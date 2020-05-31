import torch
# Timing utility
from timeit import default_timer as timer

from utils.config_parser import parse_args, parse_yaml
from dataloader import get_train_valid_dataloaders, collate_fn
from models.model_zoo import get_model
from utils.averager import Averager


def train_model(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    num_epochs,
    lr_scheduler=None,
    path_save_model='./artifacts/saved_models/fasterrcnn_test.pth'
    ):
    """
    """
    history = []

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    overall_start = timer()
    # Main loop
    for epoch in range(num_epochs):

        # Keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        # Set to training
        model.train()
        start = timer()

        for images, targets, image_ids in train_data_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            #print(f"Loss dict: {loss_dict}")
            losses = sum(loss for loss in loss_dict.values())
            #print(f"Losses: {losses}")
            loss_value = losses.item()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss_value * len(images)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"\nEpoch #{epoch}: {timer() - start:.2f} seconds elapsed.")

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode (BatchNorm and Dropout works differently)
            model.eval()
            # Validation loop
            for images, targets, image_ids in valid_data_loader:
                # Tensors to gpu
                images = list(image.to(device) for image in images)
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                loss_value = losses.item()

                valid_loss += loss_value * len(images)

        # Calculate average losses
        train_loss = train_loss / len(train_data_loader.dataaset)
        valid_loss = valid_loss / len(valid_data_loader.dataset)

        history.append([train_loss, valid_loss])

        print(
            f"\nEpoch: {epoch} \tTraining loss: {train_loss:.4f} \t"
            f"Validation loss: {valid_loss:.4f}"
        )

    # End of training
    total_time = timer() - overall_start
    print(
        f"{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} "
        "seconds per epoch"
    )

    torch.save(model.state_dict(), path_save_model)
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss']
    )

    return model, history





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

    model, history = train_model(train_data_loader, valid_data_loader, model,
        optimizer, num_epochs, lr_scheduler)

    history.to_csv('./artifacts/history/fasterrcnn_test.csv', index=False)
