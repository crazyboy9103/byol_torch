import torch
import torch.optim as optim


import wandb
from datetime import datetime

from args import args_parser
from modules import EMAHelper, AverageMeter, tsne_plot
from modules import BYOL_loss, Classifier_loss
from modules import ResNetModel, LinearClassifier
from data_utils import DataManager

if __name__ == '__main__':
    args = args_parser()

    log_writer = wandb.init(
        dir="./wandb",
        name = args.wandb_name,
        project = "byol", 
        resume = "never",
        id = datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if args.type == "pretrain":
        # networks
        online_network = ResNetModel(args, "online").to(DEVICE) 
        target_network = ResNetModel(args, "target").to(DEVICE)
        
        online_network.requires_grad_(True)
        target_network.requires_grad_(False)

        ema_helper = EMAHelper(args.epochs, args.base_tau)
        ema_helper.register(target_network)

        # optimizer
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, online_network.parameters()),
            lr = args.base_lr
        )

        # scheduler 
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=args.base_lr, verbose=True)
        # data manager
        data_manager = DataManager(args)

        # loader
        train_loader = data_manager.get_loader("train")
        test_loader = data_manager.get_loader("test")

        # Training 
        for epoch in range(args.epochs):
            train_loss = AverageMeter()

            online_network = online_network.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                orig_images, views1, views2 = map(lambda x: x.to(DEVICE, non_blocking=True), images)

                q_z1 = online_network(views1, return_prediction=True)
                q_z2 = online_network(views2, return_prediction=True)

                with torch.no_grad():
                    z1 = target_network(views1, return_projection=True)
                    z2 = target_network(views2, return_projection=True)

                    ema_helper.update(online_network, epoch)
                    ema_helper.ema(target_network)
                    
                loss = (BYOL_loss(q_z1, z1.detach()) + BYOL_loss(q_z2, z2.detach())).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss.update(loss.item())

                if batch_idx % 20 == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] train_loss : {loss.item():.2f}")


            scheduler.step()

            
            online_network = online_network.eval()
            embeddings = torch.empty((0, online_network.embedding_dim))
            classes = torch.empty((0))
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    output = online_network(images, return_embedding=True).detach().cpu()

                embeddings = torch.cat((embeddings, output))
                classes = torch.cat((classes, labels))
                
            image_name = f'./tsne/tsne_{epoch}.png'

            tsne_plot(
                embeddings,
                classes, 
                image_name,
                10 if args.dataset == "CIFAR10" else 100
            )
            wandb_log = {
                "train_loss": train_loss.avg(), 
                "tsne": [wandb.Image(image_name)],
                "epoch": epoch,
            }

            train_loss.reset()
            log_writer.log(wandb_log)


            torch.save(online_network.state_dict(), args.ckpt_path)
    
    
    elif args.type == "linear":
        online_network = ResNetModel(args, "online").to(DEVICE) 
        online_network.load_state_dict(torch.load(args.ckpt_path, DEVICE))
        online_network.requires_grad_(False)
        online_network = online_network.eval()

        linear_layer = LinearClassifier(online_network.embedding_dim, 10 if args.dataset == "CIFAR10" else 100)
        linear_layer.requires_grad_(True)
        linear_opt = optim.SGD(linear_layer.parameters(), lr=lr, momentum=0.9, nesterov=True)
        
        # data manager
        data_manager = DataManager(args)

        # loader
        train_loader = data_manager.get_loader("train")
        test_loader = data_manager.get_loader("test")
        for epoch in range(args.linear_epoch):
            train_corrs = 0
            train_total = 0
            test_corrs = 0
            test_total = 0
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.no_grad():
                    features = online_network(images, return_embedding=True)
                
                out = linear_layer(features)
                loss = Classifier_loss(out, labels)

                loss.backward()
                linear_opt.step()
                linear_opt.zero_grad()
 
                train_corrs += (out.argmax(1).eq(labels)).sum()
                train_total += len(images)

                

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    features = online_network(images, return_embedding=True)
                    out = linear_layer(features)

                    test_corrs += (out.argmax(1).eq(labels)).sum()
                    test_total += len(images)
                    
            avg_test = test_corrs.item() / test_total
            avg_train = train_corrs.item() / train_total

            print(f"Epoch [{epoch+1}/{args.linear_epoch}] train_acc : {avg_train * 100:.2f}% test_acc : {avg_test * 100:.2f}%")

            wandb_log = {
                "train_acc": avg_train,
                "test_acc": avg_test,
                "epoch": epoch,
            }
            log_writer.log(wandb_log)
        

    elif args.type == "semi":
        raise NotImplementedError
