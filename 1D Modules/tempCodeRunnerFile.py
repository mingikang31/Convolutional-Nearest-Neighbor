device = 'mps'

x_test = torch.rand(32, 12, 40).to(device)
print("Input: ", x_test.shape)

scale_factor = 4

pixel_upsample = PixelShuffle1D(scale_factor)
pixel_downsample = PixelUnshuffle1D(scale_factor)

x_up = pixel_upsample(x_test)
print("Upsampled: ", x_up.shape)

x_up_down = pixel_downsample(x_up)
print("Downsampled: ", x_up_down.shape)

if torch.all(torch.eq(x_test, x_up_down)):
    print('Inverse module works.')
