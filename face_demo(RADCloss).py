import pydiffvg
import torch
import argparse
import torch.nn
from torchvision import transforms
import cv2
import numpy as np
import ttools.modules
import clip
import torch.nn.functional as F
from torchvision import transforms
from template import imagenet_templates
import os

gamma = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:  
    return [template.format(text) for template in templates]

def clip_normalize(image, device):  
    image = F.interpolate(image, size=224, mode='bicubic').cuda()  
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(pydiffvg.get_device())
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(pydiffvg.get_device())
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    image = (image - mean) / std
    return image

def main(args):
    for svgnum in range(1):
        svgnum = svgnum + 1
        svg_path = args.svg + "img" + str(svgnum) + ".svg"
        facenames = os.listdir(args.target)
        savepath = args.results_path + "img" + str(svgnum) + "/"
        for facenum in range(1):
            facenum = facenum + 1
            results_path = savepath + str(facenum) + "/"
            target_path = args.target + str(facenum) + ".jpg"

            clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)  

            cropper = transforms.Compose([
                transforms.RandomCrop(args.crop_size),  # crop size = 400
                # transforms.ColorJitter(0, 0.5, 0.5, 0.5),
                # transforms.RandomGrayscale(p=0.25)
            ])  
            cropper2 = transforms.Compose([
                transforms.RandomCrop(500, padding=100, fill=1, padding_mode='constant')
            ])  
            cropper3 = transforms.Compose([
                transforms.RandomCrop(160, padding=35, fill=1, padding_mode='constant'),
                transforms.RandomHorizontalFlip(p=0.1) 
            ])
            augment = transforms.Compose([
                transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.1),
                transforms.Resize([512,512])
            ])  
            resizer =  transforms.Compose([
                # transforms.ColorJitter(0, 0.5, 0.5, 0.5),
                # transforms.RandomGrayscale(p=0.25),
                # transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.1),
                transforms.Resize([512, 512]),
                                           ])
           

            canvas_width, canvas_height, shapes, shape_groups = \
                pydiffvg.svg_to_scene(svg_path)  
            scene_args = pydiffvg.RenderFunction.serialize_scene( \
                canvas_width, canvas_height, shapes, shape_groups)  

            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,  # width
                         canvas_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,  # bg
                         *scene_args)  
            # The output image is in linear RGB space. Do Gamma correction before saving the image.
            pydiffvg.imwrite(img.cpu(), results_path + 'gogh1.png', gamma=gamma)  

            

            points_vars = [] 
            # print(shapes)
            for path in shapes:
                path.points.requires_grad = True
                points_vars.append(path.points)
            color_vars = {}  
            # color lock
            # print(shape_groups)
            for group in shape_groups:
                group.fill_color.requires_grad = True
                color_vars[group.fill_color.data_ptr()] = group.fill_color
            color_vars = list(color_vars.values())

            # Optimize piont:0.1-1.0 color: 0.01
            points_optim = torch.optim.Adam(points_vars, lr=0.15)  
            color_optim = torch.optim.Adam(color_vars, lr=0.00)  

            
            with torch.no_grad():
                target_img = cv2.imread(target_path)
                cv2.imwrite(results_path + "target.jpg", target_img)
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                target_img1 = target_img.transpose(2, 0, 1)   # hwc--chw

                init_img0 = cv2.imread(results_path + 'gogh1.png')
                init_img0 = cv2.cvtColor(init_img0, cv2.COLOR_BGR2RGB)
                init_img0 = init_img0.transpose(2, 0, 1)  # hwc--chw

                init_pro = []  
                img_proc4 = []  
                img_proc2 = []  
                for n in range(64):
                    b = 200
                    content_pts_path = '/home/zyx/PycharmProjects/clipimg/clipvg/apps/NBB/example1/CleanedPts/correspondence_A.txt'
                    style_pts_path = '/home/zyx/PycharmProjects/clipimg/clipvg/apps/NBB/example1/CleanedPts/correspondence_B.txt'
                    A_corres = np.loadtxt(content_pts_path, delimiter=',')
                    B_corres = np.loadtxt(style_pts_path, delimiter=',')
                    A_height = target_img1.shape[1]
                    A_width = target_img1.shape[2]
                    B_height = init_img0.shape[1]
                    B_width = init_img0.shape[2]
                    A_r, A_c = A_corres[n]
                    B_r, B_c = B_corres[n]
                    B = []
                    Arleft = int(max(0, A_r - b))
                    Arright = int(min(A_height, A_r + b))
                    Acleft = int(max(0, A_c - b))
                    Acright = int(min(A_width, A_c + b))
                    Brleft = int(max(0, B_r - b))
                    Brright = int(min(B_height, B_r + b))
                    Bcleft = int(max(0, B_c - b))
                    Bcright = int(min(B_width, B_c + b))
                    B.append(Bcleft)
                    B.append(Bcright)
                    B.append(Brleft)
                    B.append(Brright)
                    init_pro.append(B)
                    target_imag1 = target_img1[:, Arleft:Arright, Acleft:Acright]
                    tar_img1 = torch.from_numpy(target_imag1).cuda()
                    tar_img1 = resizer(tar_img1).unsqueeze(0)
                    init_svg = torch.from_numpy(init_img0[:, Brleft:Brright, Bcleft:Bcright]).cuda()
                    init_svg = resizer(init_svg).unsqueeze(0)

                    # imgaug = tar_img1.clone().squeeze(0).cpu().numpy()
                    # imgaug = imgaug.transpose(1, 2, 0)
                    # imgaug = cv2.cvtColor(imgaug, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(results_path+"tar_aug"+str(n)+".jpg", imgaug)

                    img_proc4.append(init_svg)
                    img_proc2.append(tar_img1)

                target_image = torch.from_numpy(target_img1).clone().cuda().unsqueeze(0)/255.
                init_imgsvg = torch.from_numpy(init_img0).clone().cuda().unsqueeze(0)/255.
                for n in range(32):  
                    target_crop = cropper(target_image)  
                    target_crop = augment(target_crop)  

                    # imgaug = target_crop.clone().squeeze(0).cpu().numpy()
                    # imgaug = imgaug.transpose(1, 2, 0)
                    # imgaug = cv2.cvtColor(imgaug, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(results_path+"tar_aug"+str(n+32)+".jpg", imgaug)

                    img_proc2.append(target_crop)  
                img_proc2 = torch.cat(img_proc2, dim=0)  
                img_aug2 = img_proc2

                for n in range(32): 
                    target_crop4 = cropper(init_imgsvg)  
                    target_crop4 = augment(target_crop4)  
                    img_proc4.append(target_crop4)  
                img_proc4 = torch.cat(img_proc4, dim=0)  
                img_aug4 = img_proc4

                target_features = clip_model.encode_image(clip_normalize(img_aug2, device))  
                target_features /= (target_features.clone().norm(dim=-1, keepdim=True))

               
                source = "A photo"  
                template_source = compose_text_with_templates(source, imagenet_templates)  
                tokens_source = clip.tokenize(template_source).to(pydiffvg.get_device())
                text_source = clip_model.encode_text(tokens_source).detach()
                text_source = text_source.mean(axis=0, keepdim=True)
                text_source /= text_source.norm(dim=-1, keepdim=True)  
                text_features = text_source.repeat(target_features.size(0), 1)

                svg_features = clip_model.encode_image(clip_normalize(img_aug4, device))
                svg_features /= (svg_features.clone().norm(dim=-1, keepdim=True))
                glob_direction = (target_features - text_features)
                glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)  

            for epoch in range(args.num_iter):
                print('iteration:', epoch)
                points_optim.zero_grad()
                color_optim.zero_grad()

                # Forward pass: render the image.
                scene_args = pydiffvg.RenderFunction.serialize_scene( \
                    canvas_width, canvas_height, shapes, shape_groups)  
                img = render(canvas_width,  # width
                             canvas_height,  # height
                             2,  # num_samples_x
                             2,  # num_samples_y
                             0,  # seed
                             None,  # bg
                             *scene_args)  

                # Compose img with white background  
                img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                                  device=pydiffvg.get_device()) * (
                                  1 - img[:, :, 3:4]) * 1

                # img.requires_grad = True
                # # Save the intermediate render.
                pydiffvg.imwrite(img.cpu(), results_path + 'iter_{}.png'.format(epoch), gamma=gamma)

                # img2 = img[:, :, :3]
                img3 = img.permute(2, 0, 1)  # HWC -> CHw

                
                source_image = img3
                source_img = source_image.cuda()
                source_image = source_img.clone().unsqueeze(0)

                "------------global clip loss -----------"

                img_proc = []  
                for n in range(64):
                    cur_img = img3[:, init_pro[n][2]:init_pro[n][3], init_pro[n][0]:init_pro[n][1]]
                    cur_img = resizer(cur_img)
                    cur_img = cur_img.unsqueeze(0)
                    img_proc.append(cur_img)

                for n in range(32):  
                    target_crop = cropper(source_image) 
                    target_crop = augment(target_crop)  
                    img_proc.append(target_crop)  
                img_proc = torch.cat(img_proc, dim=0)  
                img_aug = img_proc

                image_features = clip_model.encode_image(clip_normalize(img_aug, device))  
                image_features /= (image_features.clone().norm(dim=-1, keepdim=True))  

                img_direction = (image_features - svg_features)  
                img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
                loss_glob = (1 - torch.nn.functional.cosine_similarity(glob_direction, img_direction, dim=1)).mean()  

                
                loss = 500 * loss_glob
                # Backpropagate the gradients.
                print("  total loss:", loss.item(), " lpipis loss:", 0, " imgclip loss:", 500 * loss_glob.item())

                loss.backward()
                # Take a gradient descent step.
                points_optim.step()  
                color_optim.step()  
                for group in shape_groups:
                    group.fill_color.data.clamp_(0.0, 1.0)
                if epoch == 0:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 20:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 30:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 50:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 70:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 90:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 120:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 140:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 160:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 180:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 200:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                      shape_groups)
                if epoch == 220:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                  shape_groups)
                if epoch == 240:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height, shapes,
                                          shape_groups)
                if epoch == 260:
                    pydiffvg.save_svg(results_path + "{}.svg".format(epoch), canvas_width, canvas_height,
                                              shapes,
                                              shape_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path", default="/home/zyx/PycharmProjects/clipimg/clipvg/apps/imgs/")
    parser.add_argument("--target", help="target image path",
                        default="/home/zyx/PycharmProjects/clipimg/clipvg/apps/face/peterbody/")
    parser.add_argument("--results_path", help="save image path",
                        default="/home/zyx/PycharmProjects/clipimg/clipvg/apps/result_peterbody/")
    parser.add_argument("--num_iter", type=int, default=261)
    parser.add_argument('--img_size', type=int, default=512, help='size of images')
    parser.add_argument('--crop_size', type=int, default=400, help='size of images')
    parser.add_argument('--lr', type=float, default=5e-4, )
    parser.add_argument('--thresh', type=float, default=0.0, help='Number of domains')
    parser.add_argument('--num_crops', type=int, default=64, help='number of patches')
    # **********************************

    args = parser.parse_args()
    main(args)
