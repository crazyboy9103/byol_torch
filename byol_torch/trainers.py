import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy 
#import numpy as np
import os
from utils import EMAHelper

from torch.utils.data import DataLoader


def finetune_client_model(args, dataset, device, unsup_model, epochs=1, freeze=False):
    model = copy.deepcopy(unsup_model).to(device)
    
    model = model.train()
    
    if args.exp in ["SimCLR", "SimSiam", "BYOL", "FedSup"]:
        model.classifier.reset_parameters()

    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters() if not freeze else model.classifier.parameters()),
        args.lr
    )

    loader = DataLoader(
        dataset,
        batch_size = args.finetune_bs, 
        shuffle = True,
        num_workers = args.num_workers,
    )
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader): 
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images, return_logits=True)
            loss = FL_criterion(device)(preds, labels)

            loss.backward()
            optimizer.step()
            
    return copy.deepcopy(model.state_dict())

def train_server_model(args, dataset, device, sup_model, epochs=1):
    model = copy.deepcopy(sup_model).to(device)
    
    model = model.train()
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr
    )

    loader = DataLoader(
        dataset,
        batch_size = args.server_bs, 
        shuffle = True,
        num_workers = args.num_workers,
    )

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader): 
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images, return_logits=True)
            loss = FL_criterion(device)(preds, labels)

            loss.backward()
            optimizer.step()
            
    return copy.deepcopy(model.state_dict())

    

def train_client_model(args, dataset, device, sup_model = None, unsup_model = None, epochs=1, q=None, done=None, helpers=None, step=None, steps=None):
    client_model = copy.deepcopy(unsup_model).to(device)
    
    client_model = client_model.train()
    for param in client_model.parameters():
        param.requires_grad = True
    
    # this condition is same as main.py => should_send_server_model
    if args.agg == "FedSSL" or args.exp in ["FedSup", "FedMatch"]:
        assert sup_model != None, "sup model must be non null"
        ref_model = copy.deepcopy(sup_model).to(device).eval()
    
    if args.exp == "FedMatch":
        model_wrapper = copy.deepcopy(unsup_model).to(device).eval()

    if args.agg == "FedProx":
        glob_model = copy.deepcopy(unsup_model).to(device).eval()

    if args.exp == "BYOL":
        ema_helper = EMAHelper()
        ema_helper.register(client_model)
        target_net = copy.deepcopy(client_model).to(device).eval() 

    if args.exp == "FedRGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, client_model.parameters()),
            lr=args.lr,
            momentum=0.9, 
            nesterov=True,
            weight_decay=1e-4
        )
    
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, client_model.parameters()),
            lr=args.lr
        )

    loader = DataLoader(
        dataset,
        batch_size = args.local_bs, 
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True
    )

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            if args.exp == "FLSL" or args.exp == "centralized":
                orig_images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = client_model(orig_images, return_logits=True)
                loss = FL_criterion(device)(preds, labels)

            elif args.exp == "PseudoLabel":
                orig_images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    pseudo_preds = client_model(orig_images, return_logits=True)
                    pseudo_labels = pseudo_preds.argmax(dim=1)
                
                preds = client_model(orig_images, return_logits=True)
                loss = FL_criterion(device)(preds, pseudo_labels)

            elif args.exp == "FedMatch":
                orig_images, views = images
                orig_images, views = orig_images.to(device, non_blocking=True), views.to(device, non_blocking=True)

                # Consistency regularization
                with torch.no_grad():
                    psi_local = copy.deepcopy(model_wrapper.state_dict())
                    sigma_global = copy.deepcopy(ref_model.state_dict())
                    theta_local = {k: psi_local[k] + sigma_global[k] for k in psi_local}
                    model_wrapper.load_state_dict(theta_local)
                    orig_logits = model_wrapper(orig_images, return_logits=True)

                _, pseudo_labels = torch.max(orig_logits, 1) # pseudo-labels from local model

                loss = torch.tensor(0., device=device)

                if helpers != None:
                    helper_labels = {}
                    for client_id, psi_helper in helpers.items():
                        with torch.no_grad():
                            theta_local = {k: psi_helper[k] + sigma_global[k] for k in psi_helper}
                            model_wrapper.load_state_dict(theta_local)
                            helper_logits = model_wrapper(orig_images, return_logits=True)
                        
                        
                        loss += (nn.KLDivLoss(size_average="batchmean")(orig_logits, helper_logits)) / len(helpers)

                        values, indices = torch.max(nn.Softmax()(helper_logits), 1)
                        helper_label = []
                        for value, index in zip(values, indices):
                            if value > args.tau:
                                helper_label.append(index)
                            else:
                                helper_label.append(-1)

                        helper_labels[client_id] = helper_label

                for i, pseudo_label in enumerate(pseudo_labels):
                    counts = {pseudo_label: 1}
                    if helpers != None:
                        for client_id, helper_label in helper_labels.items():
                            if helper_label[i] != -1:
                                if helper_label[i] not in counts:
                                    counts[helper_label[i]] = 1
                                else:
                                    counts[helper_label[i]] += 1
                    
                    maj_vote_label = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                    pseudo_labels[i] = maj_vote_label
                             

                client_model = client_model.train()
                local_logits = client_model(views, return_logits=True)

                loss += FL_criterion(device)(local_logits, pseudo_labels)
                loss *= args.iccs_lambda

            elif args.exp in ["SimCLR", "SimSiam", "FixMatch", "BYOL", "FedSup", "FedRGD"]:
                orig_images, views1, views2 = images
                orig_images = orig_images.to(device, non_blocking=True)
                views1 = views1.to(device, non_blocking=True)
                views2 = views2.to(device, non_blocking=True)
                
                if args.exp == "SimCLR":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    loss1 = NCE_loss(device, args.temperature, p1, p2) 
                    loss2 = NCE_loss(device, args.temperature, p2, p1) 
                    loss = (loss1 + loss2).mean()
                
                elif args.exp == "SimSiam":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    loss = SimSiam_loss(device, p1, p2, z1.detach(), z2.detach())

                elif args.exp == "FixMatch":
                    # view1 = weak aug
                    # view2 = strong aug
                    client_model = client_model.eval()
                    with torch.no_grad():
                        pseudo_preds = client_model(views1, return_logits=True).detach()
                        pseudo_probs, pseudo_labels = torch.max(nn.Softmax(dim=1)(pseudo_preds), 1)
                        above_thres = pseudo_probs > args.threshold

                    client_model = client_model.train()
                    
                    if above_thres.sum().item() > 1:
                        pseudo_labels = pseudo_labels[above_thres]
                        views2 = views2[above_thres]
                        preds = client_model(views2, return_logits=True)
                        loss = FL_criterion(device)(preds, pseudo_labels)
                    else:
                        loss = torch.tensor(0., device=device)

                
                elif args.exp == "BYOL":
                    z1, p1 = client_model(views1)
                    z2, p2 = client_model(views2)
                    
                    with torch.no_grad():
                        ema_helper.update(client_model)
                        ema_helper.ema(target_net)
                        # Use EMA target net
                        ref_z1 = target_net(views1, return_embedding=True)
                        ref_z2 = target_net(views2, return_embedding=True)

                    loss1 = BYOL_loss(device, p1, ref_z2.detach())
                    loss2 = BYOL_loss(device, p2, ref_z1.detach())
                    loss = (loss1 + loss2).mean()

                elif args.exp == "FedSup":
                    if args.nocrl:
                        loss = torch.tensor(0., device=device) 
                    else:
                        z1, p1 = client_model(views1)
                        z2, p2 = client_model(views2)

                        with torch.no_grad():
                            tar_z1 = ref_model(views1, return_embedding=True)
                            tar_z2 = ref_model(views2, return_embedding=True)

                        loss1 = BYOL_loss(device, p1, tar_z2.detach())
                        loss2 = BYOL_loss(device, p2, tar_z1.detach())
                        loss = (loss1 + loss2).mean()

                # elif args.exp == "FedRGD":
                #     # view1 = weak aug
                #     # view2 = strong aug
                #     client_model = client_model.eval()
                #     with torch.no_grad():
                #         pseudo_preds = client_model(views1, return_logits=True).detach()
                #         pseudo_probs, pseudo_labels = torch.max(nn.Softmax(dim=1)(pseudo_preds), 1)
                #         above_thres = pseudo_probs > 0.95

                #     client_model = client_model.train()
                    
                #     if above_thres.sum().item() > 1:
                #         pseudo_labels = pseudo_labels[above_thres]
                #         views2 = views2[above_thres]
                #         preds = client_model(views2, return_logits=True)
                #         loss =  FL_criterion(device)(preds, pseudo_labels)

                #     else:
                #         loss = torch.tensor(0., device=device)
            
            if args.agg == "FedSSL":
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output # no detach here as we want to flow grads through this activation
                    return hook
                
                teacher_activation = {}
                def get_teacher_activation(name):
                    def hook(model, input, output):
                        teacher_activation[name] = output.detach()
                    return hook
                

                teacher_handles = []
                client_handles = []

                if args.reg_nums >= 1:
                    t_handle = ref_model.backbone.layer1.register_forward_hook(get_teacher_activation('layer1'))
                    teacher_handles.append(t_handle)

                    c_handle = client_model.backbone.layer1.register_forward_hook(get_activation('layer1'))
                    client_handles.append(c_handle)

                if args.reg_nums >= 2:
                    t_handle = ref_model.backbone.layer2.register_forward_hook(get_teacher_activation('layer2'))
                    teacher_handles.append(t_handle)

                    c_handle = client_model.backbone.layer2.register_forward_hook(get_activation('layer2'))
                    client_handles.append(c_handle)

                if args.reg_nums >= 3:
                    t_handle = ref_model.backbone.layer3.register_forward_hook(get_teacher_activation('layer3'))
                    teacher_handles.append(t_handle)
                    
                    c_handle = client_model.backbone.layer3.register_forward_hook(get_activation('layer3'))
                    client_handles.append(c_handle)

                if args.reg_nums >= 4:
                    t_handle = ref_model.backbone.layer4.register_forward_hook(get_teacher_activation('layer4'))
                    teacher_handles.append(t_handle)

                    c_handle = client_model.backbone.layer4.register_forward_hook(get_activation('layer4'))
                    client_handles.append(c_handle)
                
                mse_fn = nn.MSELoss(reduction="mean").to(device)
                dis_loss = torch.tensor(0., device=device)
                # views1 loss
                with torch.no_grad():
                    ref_z = ref_model(views1, return_embedding=True)
                orig_z = client_model(views1, return_embedding=True)

                for name in teacher_activation:
                    dis_loss += (args.mse_decay_rate ** (step / steps)) * mse_fn(teacher_activation[name], activation[name])
                
                # views2 loss
                with torch.no_grad():
                    ref_z = ref_model(views2, return_embedding=True)
                orig_z = client_model(views2, return_embedding=True)

                for name in teacher_activation:
                    dis_loss += (args.mse_decay_rate ** (step / steps)) * mse_fn(teacher_activation[name], activation[name])
                
                # avg of two losses
                dis_loss /= 2
                loss += dis_loss
                for th, ch in zip(teacher_handles, client_handles):
                    th.remove()
                    ch.remove()

            if epoch > 0:
                if args.agg == "FedProx":
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(client_model.parameters(), glob_model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    
                    loss += args.mu / 2. * w_diff

                if args.exp == "FedMatch":
                    assert args.agg != 'FedProx', 'agg must be FedAvg'
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(client_model.parameters(), ref_model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)

                    loss += args.l2_lamb * w_diff

                    w_diff_l1 = torch.tensor(0., device=device)
                    for w in client_model.parameters():
                        w_diff_l1 += torch.norm(w, 1)
                    
                    loss += args.l1_lamb * w_diff_l1

            if loss.item() == 0:
                print("Skip training as no reliable pseudo-label was generated")
                continue

            loss.backward()
            optimizer.step()

            print(f"{os.getpid()} [{batch_idx}/{len(loader)}] loss {loss.item()}")

    state_dict = copy.deepcopy(client_model.state_dict())
    
    # Multiprocessing Queue
    if q != None:
        q.put(state_dict)
        done.wait()

    return state_dict

def test_server_model(args, dataset, device, sup_model, aux_model=None):
    model = copy.deepcopy(sup_model).to(device)
    model = model.eval()
    if aux_model != None:
        a_model = copy.deepcopy(aux_model).to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size = len(dataset), 
        shuffle = True,
        num_workers = args.num_workers,
    )

    images, labels = next(iter(loader))
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images, return_logits=True)
        loss = FL_criterion(device)(preds, labels)
        loss_value = loss.item()


        _, top1_preds = torch.max(preds.data, -1)
        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
        top5 = 0
        for label, pred in zip(labels, top5_preds):
            if label in pred:
                top5 += 1

        top5 /= labels.size(0)
        top5 *= 100

    return loss_value, top1, top5

def test_client_model(args, finetune_set, test_set, device, unsup_model, finetune=True, freeze=False, finetune_epochs = 5, aux_model=None):
    # if finetune, finetune epochs must be > 0 
    assert (not finetune) or (finetune_epochs > 0 and finetune)
    
    if finetune:
        orig_state_dict = copy.deepcopy(unsup_model.state_dict())

        # load finetuned dict
        finetuned_state_dict = finetune_client_model(args, finetune_set, device, unsup_model, epochs=finetune_epochs, freeze=freeze)
        unsup_model.load_state_dict(finetuned_state_dict)

        # test
        loss, top1, top5 = test_server_model(args, test_set, device, unsup_model, aux_model = aux_model)

        # reload original state_dict
        unsup_model.load_state_dict(orig_state_dict)
        return loss, top1, top5
    
    # test using whole testset
    loss, top1, top5 = test_server_model(args, test_set, device, unsup_model, aux_model = aux_model)
    return loss, top1, top5 


def BYOL_loss(device, p, z):
    p = F.normalize(p, dim=1, p=2)
    z = F.normalize(z, dim=1, p=2)
    return 2 - 2 * (p * z).sum(dim=1)

def FL_criterion(device):
    return nn.CrossEntropyLoss(reduction="mean").to(device)

def SimSiam_loss(device, p1, p2, z1, z2):
    criterion = nn.CosineSimilarity(dim=1).to(device)
    p1, p2, z1, z2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1), F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

def NCE_loss(device, temperature, feature1, feature2, labels=None):
    feature1, feature2 = F.normalize(feature1, dim=1), F.normalize(feature2, dim=1)
    batch_size = feature1.shape[0]
    LARGE_NUM = 1e9
     
    masks = torch.eye(batch_size, device=device)
    
    logits_aa = torch.matmul(feature1, feature1.T) / temperature #similarity matrix 
    logits_aa = logits_aa - masks * LARGE_NUM
    
    logits_bb = torch.matmul(feature2, feature2.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    
    logits_ab = torch.matmul(feature1, feature2.T) / temperature
    logits_ba = torch.matmul(feature2, feature1.T) / temperature
    
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    if labels == None:
        labels = torch.arange(0, batch_size, device=device, dtype=torch.int64)
        
    loss_a = criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
    
    loss = loss_a + loss_b
    return loss / 2
