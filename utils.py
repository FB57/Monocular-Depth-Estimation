import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from layers import BilinearUpSampling2D
from huggingface_hub import from_pretrained_keras


def depth_norm(x, maxDepth):
    return maxDepth / x

def infer(image, model):
    print(np.shape(image))
    inputs = load_images([image])
    outputs = predict(model, inputs)
    print(f"Input shape: {np.shape(image)}; Output shape: {np.shape(outputs)}")
    
    # plasma = plt.get_cmap('plasma')
    rescaled = outputs[0][:, :, 0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled / np.max(rescaled)
    # image_out = plasma(rescaled)[:, :, :3]
    # return image_out
    return rescaled

def box_blur_variance(ksize):
    #To calculate variance based on kernel size
    x = np.arange(ksize) - ksize // 2
    x, y = np.meshgrid(x, x)
    return np.mean(x**2 + y**2)

def var_blur(img, depthmap, sigma=7, ksize=3):
    # n_c = img.shape[2]
    sigma = sigma*depthmap
    variance = [box_blur_variance(ksize)]

    # Number of times to blur per-pixel
    num_box_blurs = 2 * sigma**2 / variance

    # Number of rounds of blurring
    max_blurs = int(np.ceil(np.max(num_box_blurs))) * 3

    # Approximate blurring a variable number of times
    blur_weight = num_box_blurs / max_blurs

    current_im = np.copy(img)
    for i in range(max_blurs):
        next_im = cv.blur(current_im, (ksize, ksize))
        # print(next_im.shape)
        current_im = next_im * blur_weight + current_im* (1 - blur_weight)
        # current_im[..., 0] = next_im[..., 0] * blur_weight + current_im[..., 0]* (1 - blur_weight)
        # current_im[..., 1] = next_im[..., 1] * blur_weight + current_im[..., 1] * (1 - blur_weight)
        # current_im[..., 2] = next_im[..., 2] * blur_weight + current_im[..., 2] * (1 - blur_weight)
    # current_im = Image.fromarray(current_im)
    # print(np.shape(current_im))
    # print("saving Image as current_im.jpeg")
    # current_im.save('current_im.jpeg')
    return current_im


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(file.reshape(480, 640, 3) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def test(img_gray, disparity):
    f, axxarr = plt.subplots(3, 3)
    f.set_figwidth(16)
    f.set_figheight(10)

    si = 0
    ki = 0
    for s in [5, 10, 20]:
        ki = 0
        for k in [3, 7, 13]:
            bokeh = var_blur(img_gray, disparity, s, k)
            axxarr[si, ki].imshow(bokeh, cmap="gray")
            # axxarr[si, ki].xlabel(f"Sigma = {s}; Kernel Size = {(k, k)}")
            ki+=1
        si+=1

def show_results(img, bokeh):
    f, axxarr = plt.subplots(1, 2)
    f.set_figwidth(16)
    f.set_figheight(10)

    axxarr[0].imshow(img)
    axxarr[1].imshow(bokeh)

    plt.show()

def get_model():
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    print('Loading model...')
    model = from_pretrained_keras("keras-io/monocular-depth-estimation", custom_objects=custom_objects, compile=False)

    # if load_model:
    #     model = from_pretrained_keras("model/models--keras-io--monocular-depth-estimation", custom_objects=custom_objects, compile=False, local_files_only=True)
    # if save_model:
    #     model = from_pretrained_keras("keras-io/monocular-depth-estimation", custom_objects=custom_objects, compile=False, cache_dir="model/")
   
    # if save_model:
    #     model.save_pretrained("./model/") 

    print('Successfully loaded model...')
    return model

def pre_proc(img, sigma):
    model = get_model()
    #RESIZING IMAGE TO FIT MODEL INPUT
    img = cv.resize(img, (640, 480),
                interpolation = cv.INTER_LINEAR)

    #GENERATING DISPARITY MAP USING MODEL
    disparity = infer(img, model)
    disparity = cv.resize(disparity, (640, 480),
                interpolation = cv.INTER_LINEAR)        #UPSCALING TO MATCH IMAGE SIZE

    return bokeh_gen(img, disparity, sigma)
    # return img, disparity

def bokeh_gen(img, disparity, sigma):
    bokeh_r = var_blur(img[..., 0], disparity, sigma=sigma)
    bokeh_g = var_blur(img[..., 1], disparity, sigma=sigma)
    bokeh_b = var_blur(img[..., 2], disparity, sigma=sigma)
    bokeh = np.dstack((bokeh_r, bokeh_g, bokeh_b))/255

    return bokeh

def launch_interface():
    import gradio as gr
    # with gr.Blocks(title="Monocular Depth Estimation") as demo:
    #     gr.Markdown("Keras Implementation of Unet architecture with Densenet201 backbone for estimating the depth of image 📏")
    #     with gr.Row():
    #         with gr.Column():
    #             input = gr.Image(label="image", type="numpy")
    #             disparity = gr.Image(label="disparity map", image_mode="L")
    #             sigma = gr.Slider(0, 50, value=7, step=1)
    #         output = gr.Image(label="output", shape=(640,480))
    #     new_img = gr.Image(shape=(640, 480), visible=False)
    #     input.change(pre_proc, input, outputs=[new_img, disparity], show_progress=True)
    #     btn = gr.Button("Run").style(full_width=False)
    #     btn.click(fn=bokeh_gen, inputs=[new_img, disparity, sigma], outputs=output)
    #     sigma.change(fn=bokeh_gen, inputs=[new_img, disparity, sigma], outputs=output)
    # demo.launch()
    iface = gr.Interface(
    fn=pre_proc,
    title="Monocular Bokeh Estimation",
    description = "Keras Implementation of Unet architecture with Densenet201 backbone for estimating the depth of image 📏",
    inputs=[gr.inputs.Image(label="image", type="numpy"),
            gr.Slider(0, 50, value=7, label="sigma")],
    outputs="image",
    cache_examples=False).launch(debug=True)