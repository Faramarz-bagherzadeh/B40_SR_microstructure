import kornia.augmentation as K
import torch.nn as nn

# input data should have one extra dimenssion for batch 
class MyAugmentationPipeline(nn.Module):
   def __init__(self) -> None:
      super(MyAugmentationPipeline, self).__init__()
      self.DF = K.RandomDepthicalFlip3D(p=0.1,same_on_batch = True,keepdim = True)
      self.HF = K.RandomHorizontalFlip3D(p=0.1,same_on_batch = True,keepdim = True)
      self.VF = K.RandomVerticalFlip3D(p=0.1,same_on_batch = True,keepdim = True)
      #self.RO = K.RandomRotation3D((10., 10., 10.), p=0.7, same_on_batch=True, keepdim=True)
    
   def forward(self, input, mask):

      DF_params = self.DF.forward_parameters(input.shape)
      input = self.DF(input, DF_params)
      mask = self.DF(mask, DF_params)
      
      HF_params = self.HF.forward_parameters(input.shape)
      input = self.HF(input, HF_params)
      mask = self.HF(mask, HF_params)
      
      VF_params = self.VF.forward_parameters(input.shape)
      input = self.VF(input, VF_params)
      mask = self.VF(mask, VF_params)
      
      #RO_params = self.RO.forward_parameters(input.shape)
      #input = self.RO(input, RO_params)
      #mask = self.RO(mask, RO_params)


      return input, mask
  
  
  
  
'''
        def Random__gamma(img,gamma):
            gamma = random.randint(70, 130) * gamma
            print (gamma)
            image_float = img.astype(float)
            corrected_image = np.power(image_float, gamma)
            corrected_image = (corrected_image * 255).astype(np.uint8)
            return corrected_image
        
        
        def Random_Bluring_3D( img,sigma):

            sigma = random.randint(10, 20) * sigma
            blured = skimage.filters.gaussian(img,sigma=sigma,preserve_range=True)
            blured = blured.astype('uint8')
            blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return blured
        
        
        def Random_speckle_noise_3D(image, sigma):
            sigma = random.randint(10, 20) * sigma
            # Generate random Gaussian noise
            depth, height, width = image.shape
            noise = np.random.randn(depth, height, width) * sigma

            # Add the noise to the image
            noisy_image = image + image * noise
        
            # Normalize the image to [0, 255]
            noisy_image = cv2.normalize(noisy_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            return noisy_image
'''