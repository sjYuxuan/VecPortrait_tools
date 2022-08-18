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

"--------------------------------------------一些函数工具------------------------------------------------"
def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:  # 箭头和冒号是类型提示，表示输出输入的类型为什么
    return [template.format(text) for template in templates]

def clip_normalize(image, device):  # 归一化
    image = F.interpolate(image, size=224, mode='bicubic').cuda()  # 对图像上下采样，这里是采样到224*224
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(pydiffvg.get_device())
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(pydiffvg.get_device())
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    image = (image - mean) / std
    return image

def main(args):
    for svgnum in range(1):
        svgnum = svgnum + 1
        svg_path = args.svg + "peterbody" + str(svgnum) + ".svg"
        facenames = os.listdir(args.target)
        savepath = args.results_path + "man" + str(svgnum) + "/"
        for facenum in range(len(facenames)):
            facenum = facenum + 1
            results_path = savepath + str(facenum) + "/"
            target_path = args.target + str(facenum) + ".jpg"

            "--------------------用于计算clip-----------------"
            clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)  # 加载vit预训练模型

            "------------------------------数据剪裁等图处理部分--------------------------------"
            cropper = transforms.Compose([
                transforms.RandomCrop(args.crop_size),  # crop size = 400
                # transforms.ColorJitter(0, 0.5, 0.5, 0.5),
                # transforms.RandomGrayscale(p=0.25)
            ])  # 随机剪裁输出尺寸400*400
            cropper2 = transforms.Compose([
                transforms.RandomCrop(500, padding=100, fill=1, padding_mode='constant')
            ])  # 随机剪裁，128*128 255像素值填充
            cropper3 = transforms.Compose([
                transforms.RandomCrop(160, padding=35, fill=1, padding_mode='constant'),
                transforms.RandomHorizontalFlip(p=0.1)  # 以0.3的概率随机水平翻转
            ])
            augment = transforms.Compose([
                transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.1),
                transforms.Resize([512,512])
            ])  # 以1的概率进行0.3变形程度的透视变换
            resizer =  transforms.Compose([
                # transforms.ColorJitter(0, 0.5, 0.5, 0.5),
                # transforms.RandomGrayscale(p=0.25),
                # transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.1),
                transforms.Resize([512, 512]),
                                           ])
            "****************************加载源（svg-img）图像*********************************"

            canvas_width, canvas_height, shapes, shape_groups = \
                pydiffvg.svg_to_scene(svg_path)  # svg加载并转换为tensor
            scene_args = pydiffvg.RenderFunction.serialize_scene( \
                canvas_width, canvas_height, shapes, shape_groups)  # 给定一组形状，将它们转换为pytorch可使用参数的线性列表。

            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,  # width
                         canvas_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,  # bg
                         *scene_args)  # 将图片光栅化
            # The output image is in linear RGB space. Do Gamma correction before saving the image.
            pydiffvg.imwrite(img.cpu(), results_path + 'gogh1.png', gamma=gamma)  # 保存光栅化后的图像

            "*********************************设置优化参数与优化器*************************************"

            points_vars = []  # 保存点的值用于参数更新
            # print(shapes)
            for path in shapes:
                path.points.requires_grad = True
                points_vars.append(path.points)
            color_vars = {}  # 保存颜色的值用于参数更新
            # color lock
            # print(shape_groups)
            for group in shape_groups:
                group.fill_color.requires_grad = True
                color_vars[group.fill_color.data_ptr()] = group.fill_color
            color_vars = list(color_vars.values())

            # Optimize piont:0.1-1.0 color: 0.01
            points_optim = torch.optim.Adam(points_vars, lr=0.15)  # 点的优化器（形状）
            color_optim = torch.optim.Adam(color_vars, lr=0.00)  # 颜色优化器

            "*********************************开始迭代**********************************"
            with torch.no_grad():
                target_img = cv2.imread(target_path)
                cv2.imwrite(results_path + "target.jpg", target_img)
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                target_img1 = target_img.transpose(2, 0, 1)   # hwc--chw

                init_img0 = cv2.imread(results_path + 'gogh1.png')
                init_img0 = cv2.cvtColor(init_img0, cv2.COLOR_BGR2RGB)
                init_img0 = init_img0.transpose(2, 0, 1)  # hwc--chw

                init_pro = []  # 用于保存截取框的下标
                img_proc4 = []  # 保存svg图像数据增广后的图片列表
                img_proc2 = []  # 保存target图像数据增广后的图片列表
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
                for n in range(32):  # num_crops列表长度，即n次剪裁后保存的n张图片
                    target_crop = cropper(target_image)  # 之前定义的图片剪裁
                    target_crop = augment(target_crop)  # 透视变换

                    # imgaug = target_crop.clone().squeeze(0).cpu().numpy()
                    # imgaug = imgaug.transpose(1, 2, 0)
                    # imgaug = cv2.cvtColor(imgaug, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(results_path+"tar_aug"+str(n+32)+".jpg", imgaug)

                    img_proc2.append(target_crop)  # 将处理过的图片加到列表中
                img_proc2 = torch.cat(img_proc2, dim=0)  # 将图片按batch维度拼接
                img_aug2 = img_proc2

                for n in range(32):  # 32张全貌增强图
                    target_crop4 = cropper(init_imgsvg)  # 之前定义的图片剪裁
                    target_crop4 = augment(target_crop4)  # 透视变换
                    img_proc4.append(target_crop4)  # 将处理过的图片加到列表中
                img_proc4 = torch.cat(img_proc4, dim=0)  # 将图片按batch维度拼接
                img_aug4 = img_proc4

                target_features = clip_model.encode_image(clip_normalize(img_aug2, device))  # 目标图像编码
                target_features /= (target_features.clone().norm(dim=-1, keepdim=True))

                "*****************************得到源文本（a photo）的特征向量（1*n）**********************************"
                source = "A photo"  # 源文本
                template_source = compose_text_with_templates(source, imagenet_templates)  # 将a photo与文本模板合成列表
                tokens_source = clip.tokenize(template_source).to(pydiffvg.get_device())
                text_source = clip_model.encode_text(tokens_source).detach()
                text_source = text_source.mean(axis=0, keepdim=True)
                text_source /= text_source.norm(dim=-1, keepdim=True)  # 得到源文本的特征向量
                text_features = text_source.repeat(target_features.size(0), 1)

                svg_features = clip_model.encode_image(clip_normalize(img_aug4, device))
                svg_features /= (svg_features.clone().norm(dim=-1, keepdim=True))
                glob_direction = (target_features - text_features)
                glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)  # 图片方向

            for epoch in range(args.num_iter):
                print('iteration:', epoch)
                points_optim.zero_grad()
                color_optim.zero_grad()

                # Forward pass: render the image.
                scene_args = pydiffvg.RenderFunction.serialize_scene( \
                    canvas_width, canvas_height, shapes, shape_groups)  # 转为pytorch可用的参数线性表
                img = render(canvas_width,  # width
                             canvas_height,  # height
                             2,  # num_samples_x
                             2,  # num_samples_y
                             0,  # seed
                             None,  # bg
                             *scene_args)  # 转为图片  *表示收集剩余的参数  图片为hwc，rgba

                # Compose img with white background  RGBA--RGB且与背景白色相融合
                img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                                  device=pydiffvg.get_device()) * (
                                  1 - img[:, :, 3:4]) * 1

                # img.requires_grad = True
                # # Save the intermediate render.保存渲染图
                # pydiffvg.imwrite(img.cpu(), results_path + 'iter_{}.png'.format(epoch), gamma=gamma)

                # img2 = img[:, :, :3]
                img3 = img.permute(2, 0, 1)  # HWC -> CHw

                "*********************************用于计算clip损失***************************************"
                source_image = img3
                source_img = source_image.cuda()
                source_image = source_img.clone().unsqueeze(0)

                "------------global clip loss -----------"

                img_proc = []  # 保存当前图像数据增广后的图片列表
                for n in range(64):
                    cur_img = img3[:, init_pro[n][2]:init_pro[n][3], init_pro[n][0]:init_pro[n][1]]
                    cur_img = resizer(cur_img)
                    cur_img = cur_img.unsqueeze(0)
                    img_proc.append(cur_img)

                for n in range(32):  # num_crops列表长度，即n次剪裁后保存的n张图片
                    target_crop = cropper(source_image)  # 之前定义的图片剪裁
                    target_crop = augment(target_crop)  # 透视变换
                    img_proc.append(target_crop)  # 将处理过的图片加到列表中
                img_proc = torch.cat(img_proc, dim=0)  # 将图片按batch维度拼接
                img_aug = img_proc

                image_features = clip_model.encode_image(clip_normalize(img_aug, device))  # 当前img输入图像编码
                image_features /= (image_features.clone().norm(dim=-1, keepdim=True))  # 得到图像特征二维向量

                img_direction = (image_features - svg_features)  # 图像特征迁移方向的差距
                img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
                loss_glob = (1 - torch.nn.functional.cosine_similarity(glob_direction, img_direction, dim=1)).mean()  # 全局损失

                '--------求和--------'
                loss = 500 * loss_glob
                # Backpropagate the gradients.
                print("  total loss:", loss.item(), " lpipis loss:", 0, " imgclip loss:", 500 * loss_glob.item())

                loss.backward()
                # Take a gradient descent step.
                points_optim.step()  # 更新点的参数
                color_optim.step()  # 更新颜色的参数
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