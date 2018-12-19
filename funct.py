import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array # For Converting Image to array
from tensorflow.python.keras import models # For making model
from PIL import Image
import matplotlib.pyplot as plt
import IPython.display
import tensorflow.contrib.eager as tfe
import time

def load_image(path):
    """
    Loads image for viewing & resizes them
    Returns Np array of size (1,h,w,ch)
    """
    max_dim=512 # To make a perfect view
    img = Image.open(path) # Opening image
    print(img.size)
    long = max(img.size)  # Max Dimension of image
    scale = max_dim/long  # scale is made so that image can be scaled down in both directions appropriately
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = img_to_array(img) # For converting image to array
    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img):
    """
    Shows Image Output
    """
    plt.imshow(img[0].astype('uint8'))

def process_img(path):
    """
    Processes Image by adding means of VGG19 Network Requirement
    Uses load_image function to read & process image in size
    """
    img=load_image(path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(img):
    x=img.copy() # Makes of Copy of Numpy Array
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
        
    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    # Adds Values of Means Back so that image can be made for viewing.
    # Clips Data
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def get_model(content_layers,style_layers):
    """ Creates our model with access to intermediate layers. 
      
      This function will load the VGG19 model and access the intermediate layers. 
      These layers will then be used to create a new model that will take input image
      and return the outputs from these intermediate layers from the VGG model. 
      Returns:
          returns a keras model that takes image inputs and outputs the style and 
          content intermediate layers. 
      """
      # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers] # Selects Style Outputs
    content_outputs = [vgg.get_layer(name).output for name in content_layers] # Selects Content Outputs
    model_outputs = style_outputs + content_outputs
    # Build model with inputs as vgg inputs & outputs as model_outputs
    return models.Model(vgg.input, model_outputs)

def get_content_loss(pastiche_layer,content_layer,alpha):
    """
    Returns Content Loss when Given Args. with alpha Constant 
    And Activations of a particular layer of style & content
    """
    return tf.reduce_mean(tf.square(tf.subtract(tf.multiply(alpha,pastiche_layer),tf.multiply(alpha,content_layer))))

""" For style loss, we need to calc gram matrix for each layer"""
def gram_matrix(layer):
    # We make the image channels first 
    channels = int(layer.shape[-1])
    a = tf.reshape(layer, [-1, channels])
    print(a.shape)
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target,beta):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(tf.multiply(beta,gram_style) - tf.multiply(beta,gram_target)))

def get_representations(model,content_path,style_path,num_content_layers,num_style_layers):
    """ this function computes the content layers & style layers
    of content & style image respectively.
    Args :
    model-> Pretrained vgg19 model
    content_path-> path to content Image
    style_path-> path to style image
    content_layers-> name of content_layers
    style_layers-> name of style_layers
    Returns:
    style_features & content_features
    """
    # Loading Images
    content_image=process_img(content_path)
    style_image=process_img(style_path)
    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    
    return style_features, content_features


def compute_loss(model,loss_weights,init_image, gram_features, content_features,num_content_layers,num_style_layers):
    """ This function computes total loss
    Args :
        model-> to calc layers on init_image
        alpha -> constant
        beta -> constant
        gram_features: Precomputed gram matrices corresponding to the 
        defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of 
        interest.
    Returns :
        total_loss
        style_loss
        content_loss
        """
    (alpha,beta)=loss_weights
    model_outputs=model(init_image)
    style_output_features=model_outputs[:num_style_layers]
    content_output_features=model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0
    # Accumulate Losses from all layers
    # Weighing loss from every layer Equally
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style,init_style in zip(gram_features,style_output_features):
        style_score+=weight_per_style_layer*get_style_loss(init_style[0],target_style,beta)
    
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content,alpha)
    
    loss=content_score+style_score
    return loss,content_score,style_score

def compute_grads(cfg):   # Computes gradient using tf.GradientTape
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
# Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path,style_path,content_layers,style_layers,num_iterations=1000,
                       alpha=1e3,beta=1e-2): 
    # We don't need to (or want to) train any layers of our model, so we set their trainable to false. 
    model = get_model(content_layers,style_layers)
    for layer in model.layers:
        layer.trainable = False
    loss_weights = (beta, alpha)
    # Get the style and content feature representations (from our specified intermediate layers) 
    style_features, content_features = get_representations(model, content_path, style_path,
                                                                   len(content_layers), len(style_layers))
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    # Set initial image
    init_image = process_img(content_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    
    # For displaying intermediate images 
    iter_count = 1
    # Store our best result
    best_loss, best_img = float('inf'), None
    # Create a nice config 
    loss_weights = (alpha, beta)
    #,num_content_layers,num_style_layers
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_features': gram_style_features,
        'content_features': content_features,
        'num_content_layers':len(content_layers),
        'num_style_layers':len(style_layers)
    }
    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss,content_score,style_score=all_loss
        optimizer.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time() 
    
    if loss < best_loss:
        # Update best loss and best image from total loss. 
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
        start_time = time.time()
      
        # Use the .numpy() method to get the concrete numpy array
        plot_img = init_image.numpy()
        plot_img = deprocess_img(plot_img)
        imgs.append(plot_img)
        IPython.display.clear_output(wait=True)
        IPython.display.display_png(Image.fromarray(plot_img))
        print('Iteration: {}'.format(i))        
        print('Total loss: {:.4e}, ' 
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    return best_img, best_loss 

def image_to_array(arr):
    Image.fromarray(arr)
    
def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_image(content_path) 
    style = load_image(style_path)
    plt.subplot(1, 2, 1)
    imshow(content)
    plt.subplot(1, 2, 2)
    imshow(style)

    if show_large_final: 
        plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
    

