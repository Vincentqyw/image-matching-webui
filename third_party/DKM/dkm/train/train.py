from tqdm import tqdm
from dkm.utils.utils import to_cuda


def train_step(train_batch, model, objective, optimizer, **kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)
    l.backward()
    optimizer.step()
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, progress_bar=True
):
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n=n,
        )
        lr_scheduler.step()


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
