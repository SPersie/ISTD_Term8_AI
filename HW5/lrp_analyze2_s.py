# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


import PIL
import matplotlib.pyplot as plt
import numpy as np
import os




def load_image(path, size):


    #need here ret= np array with dtype np.float32 from PIL Image 
    #after the pil image was resized to (size,size)

    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    if ret.ndim == 2:
        ret.resize((size, size, 1))
        ret = np.repeat(ret, 3, axis=-1)
    return ret




if __name__ == "__main__":
    # Load an image.
    # Need to download example images


    ###############################################################################
    ###############################################################################
    ###############################################################################
    #your code here
    fn= 'images/sudi.jpg'# your image here #os.path.join('/mnt/scratch1/data/imagespart/','ILSVRC2012_val_00000006.JPEG')
    ###############################################################################
    ###############################################################################
    ###############################################################################


    import innvestigate
    import innvestigate.utils

    import keras.backend
    

    modelind=1
    analyzerind=0

    models0=['vgg16','dense121','inception_v3']
    analyzers0=['lrp.sequential_preset_a_flat', 'gradient', 'guided_backprop']

    model=models0[modelind]
    sanalyzer=analyzers0[analyzerind]

    if model == 'vgg16':
      import keras.applications.vgg16 as vgg16
      #Get model
      model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
      size=224
    elif model == 'dense121':
       
      import keras.applications.densenet as nnnnet
      model, preprocess = nnnnet.DenseNet121(), nnnnet.preprocess_input 
      size=224
    elif model == 'inception_v3':
      import keras.applications.inception_v3 as inception_v3
      model, preprocess = inception_v3.InceptionV3(), inception_v3.preprocess_input
      size=299
 
    else:
      print('err')
      exit()

    image = load_image(fn, size) 

    
    # Code snippet.
    plt.imshow(image/255)
    plt.axis('off')
    plt.savefig("readme_example_input.png")


    # Add batch axis and preprocess
    x = preprocess(image[None],backend=keras.backend) # fix for resnets



    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    #analyzer = innvestigate.create_analyzer("lrp.sequential_preset_b_flat", model)
    if sanalyzer == 'lrp.sequential_preset_a_flat':
      analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat", model)
    elif sanalyzer == 'gradient':
      analyzer = innvestigate.create_analyzer("gradient", model)
    elif sanalyzer == 'guided_backprop':
      analyzer = innvestigate.create_analyzer("guided_backprop", model)
    else:
      print('err2')
      exit()

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    # Plot
    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    plt.savefig("readme_example_analysis.png")

    

