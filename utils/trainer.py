import torch
from torch import nn
from tqdm import tqdm
from .loss import ContentLoss, AdversialLoss
from torch.nn import functional as F

def contentPretrain(
        model,
        train_laoder,
        learning_rate = 1e-3,
        beta1 = .5, beta2 = .99,
        weight_decay = 1e-3,
        epochs = 10,
        show_steps = 1,
        device = 'cuda:0'
) -> None:
    torch.manual_seed(42)
    # set model
    model.train()
    model.to(device)

    # optimizer
    opt = torch.optim.AdamW(model.parameters(),lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay)
    
    # loss function
    content_loss = ContentLoss().to(device)

    # training
    
    for epoch in range(epochs):
        Gloss = []
        with tqdm(
        total=len(train_laoder),
        desc="Epoch progress",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
            for _, data in enumerate(train_laoder):
                data = data.to(device)
                target = model(data)
                loss = content_loss(target,data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                Gloss.append(loss.detach().cpu().item())
                pbar.update(1)

            if (epoch + 1) % show_steps == 0:
                print("Epoch [{}/{} {:.2f}%]: content loss: {:.4f}".format(
                    epoch+1, epochs, (epoch+1)/epochs * 100, sum(Gloss)/len(Gloss)
                ))
            

    model.eval()

def trainWithoutEdge(
        generator, discriminator,
        content_laoder, style_loader,
        learning_rate = 1e-3,
        beta1 = .5, beta2 = .99,
        weight_decay = 1e-3,
        epochs = 10,
        show_steps = 1,
        device = 'cuda:0'
):
    torch.manual_seed(42)

    generator.train()
    generator.to(device)
    discriminator.train()
    discriminator.to(device)

    optG = torch.optim.AdamW(generator.parameters(),lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay)
    optD = torch.optim.AdamW(discriminator.parameters(),lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    # loss functions
    content_loss = ContentLoss(omega=1).to(device)

    # begin training
    for epoch in range(epochs):
        Dloss, Gloss_recon, Gloss_adv = [], [], []
        # optimize generator
        for _, content_img in enumerate(content_laoder):
            content_img = content_img.to(device)
            target_img = generator(content_img)
            d_target = discriminator(target_img)
            gen_loss_adv = -torch.log(d_target+1e-10).mean()
            gen_loss_recon = content_loss(target_img,content_img)
            gen_loss = gen_loss_adv + gen_loss_recon
            optG.zero_grad()
            gen_loss.backward()
            optG.step()
            Gloss_recon.append(gen_loss_recon)
            Gloss_adv.append(gen_loss_adv)

        # optimize discriminator
        for _, (style_img,content_img) in enumerate(zip(style_loader,content_laoder)):
            content_img, style_img = content_img.to(device), style_img.to(device)
            d_style = discriminator(style_img)
            d_target = discriminator(generator(content_img))
            dis_loss = (-torch.log(torch.abs(1-d_target)+1e-10).mean() - torch.log(d_style+1e-10).mean())
            optD.zero_grad()
            dis_loss.backward()
            optD.step()

            Dloss.append(dis_loss)
        
        if (epoch+1)%show_steps == 0:
            print("Epoch[{}/{} {:.2f}%]: Gen_loss_recon: {:.4f}; Gen_loss_adv: {:.4f}; Dis_loss: {:.4f}".format(
                epoch+1, epochs, (epoch+1)/epochs * 100, 
                sum(Gloss_recon)/len(Gloss_recon), sum(Gloss_adv)/len(Gloss_adv),
                sum(Dloss)/len(Dloss) 
            ))
    generator.eval()
    discriminator.eval()


def trainWithoutEdge_(
        generator, discriminator,
        content_laoder, style_loader,
        batch_size = 16,
        image_size = 256,
        learning_rate = 1e-3,
        beta1 = .5, beta2 = .99,
        weight_decay = 1e-3,
        epochs = 10,
        show_steps = 1,
        device = 'cuda:0'
):

    generator.train()
    generator.to(device)
    discriminator.train()
    discriminator.to(device)

    optG = torch.optim.AdamW(generator.parameters(),lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay)
    optD = torch.optim.AdamW(discriminator.parameters(),lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay)
    
    # loss functions
    content_loss = ContentLoss().to(device)
    bce_loss = nn.BCELoss().to(device)

    style_labels = torch.ones (batch_size, 1, image_size // 4, image_size // 4).to(device)
    fake_labels    = torch.zeros(batch_size, 1, image_size // 4, image_size // 4).to(device)

    # begin training
    for epoch in range(epochs):
        Dloss, Gloss_recon, Gloss_adv = [], [], []
        with tqdm(
        total=len(content_laoder),
        desc="Epoch progress",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
            for j, content_img in enumerate(content_laoder):
                # optimize discriminator
                content_img = content_img.to(device)
                for i, style_img in enumerate(style_loader):
                    style_img = style_img.to(device)
                    d_style = discriminator(style_img)
                    d_target = discriminator(generator(content_img))
                    dis_loss = bce_loss(d_style,style_labels[:d_style.size()[0]]) + bce_loss(d_target,fake_labels[:d_target.size()[0]])
                    optD.zero_grad()
                    dis_loss.backward()
                    optD.step()
                    Dloss.append(dis_loss)

                    # optimize generator
                    target = generator(content_img)
                    d_target = discriminator(target)
                    gen_adv = bce_loss(d_target,style_labels[:d_target.size()[0]])
                    gen_recon = content_loss(target,content_img)
                    gen_loss = gen_adv+gen_recon
                    optG.zero_grad()
                    gen_loss.backward()
                    optG.step()
                    Gloss_recon.append(gen_recon)
                    Gloss_adv.append(gen_adv)
                    # if i>=50:
                    #     break
                pbar.update(1)
                if j >= 99:
                    break
                    
        
        if (epoch+1)%show_steps == 0:
            print("Epoch[{}/{} {:.2f}%]: Gen_loss_recon: {:.4f}; Gen_loss_adv: {:.4f}; Dis_loss: {:.4f}".format(
                epoch+1, epochs, (epoch+1)/epochs * 100, 
                sum(Gloss_recon)/len(Gloss_recon), sum(Gloss_adv)/len(Gloss_adv),
                sum(Dloss)/len(Dloss) 
            ))
    generator.eval()
    discriminator.eval()

            

