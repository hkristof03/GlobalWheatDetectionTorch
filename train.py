import torch

from utils.config_parser import parse_args, parse_yaml
from dataloader import get_train_valid_dataloaders, collate_fn
from models.model_zoo import get_model
from utils.averager import Averager


def train_model(
    train_data_loader,
    model,
    optimizer,
    num_epochs,
    lr_scheduler=None
    ):
    """
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):

        loss_hist.reset()

        for images, targets, image_ids in train_data_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            print(f"Loss dict: {loss_dict}")
            losses = sum(loss for loss in loss_dict.values())
            print(f"Losses: {losses}")
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")


if __name__ == '__main__':

    args = parse_args()
    configs = parse_yaml(args.pyaml)
    configs_dataloader = configs['dataloader']

    train_data_loader, valid_data_loader = get_train_valid_dataloaders(
        configs_dataloader,
        collate_fn
    )
    model = get_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
        weight_decay=0.0005)
    lr_scheduler = None
    num_epochs = 10

    train_model(train_data_loader, model, optimizer, num_epochs, lr_scheduler)
