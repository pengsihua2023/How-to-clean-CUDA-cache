# How-to-clean-CUDA-cache
```
class MyModel(LightningModule):
    def __init__(self, model, train_dataset, test_dataset, batch_size=1, lr=1e-5, accumulation_steps=20):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, except 'model'
        self.automatic_optimization = False  # Disable automatic optimization
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.accumulation_steps = accumulation_steps  # New attribute for gradient accumulation steps

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Gradient accumulation
        loss = loss / self.accumulation_steps  # Scale loss
        self.manual_backward(loss)  # Backward pass (accumulates gradients)
        
        # Condition to check if it is time to update weights
        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader()):
            self.optimizer.step()  # Update weights
            self.optimizer.zero_grad()  # Clear gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:   
                torch.cuda.empty_cache()    #clean Cuda cache
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)  # Save optimizer as attribute
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8)

```
