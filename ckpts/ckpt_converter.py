import torch
import pytorch_lightning


def convert_torch_to_lightning(torch_checkpoint_path, lightning_checkpoint_path, last_epoch, last_global_step):
    # Load the regular PyTorch checkpoint
    torch_checkpoint = torch.load(torch_checkpoint_path, map_location='cpu')

    model_state_dict = {k.replace('module.', ''): v for k, v in torch_checkpoint['model_state_dict'].items()}

    # Convert to the PyTorch Lightning checkpoint format
    lightning_checkpoint = {
        'epoch': last_epoch,
        'global_step': last_global_step,
        'pytorch-lightning_version': pytorch_lightning.__version__,
        'state_dict': model_state_dict,
        'optimizer_states': [torch_checkpoint['optimizer_state_dict']],
        'lr_schedulers': [],  # Assuming no LR schedulers for simplicity. Adjust as necessary.
    }

    # Save the converted checkpoint in Lightning format
    torch.save(lightning_checkpoint, lightning_checkpoint_path)


if __name__ == "__main__":
    # model = torch.load("/media/mldadmin/home/s123mdg35_05/ar-ldm/ckpts/flintstones_repro_v2/last.ckpt")
    # model_vanilla = torch.load("/media/mldadmin/home/s123mdg35_05/ar-ldm/ckpts/flintstones_official_ckpt/flintstones.pth")
    src_path = r'D:\AR-LDM\ckpts\flintstones_official_ckpt\flintstones.pth'
    dst_path = r'D:\AR-LDM\ckpts\flintstones_official_ckpt\flintstones.ckpt'
    last_epoch = 0
    last_global_step = 0
    convert_torch_to_lightning(src_path, dst_path, last_epoch, last_global_step)
