import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import func.depart_freq as DeFreq
import func.enhance_img as Enhimg
import torchvision.transforms as transforms

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,#用UNet
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None,xh=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
                    
            x0_tmp=(x+1)/2.0*255.0
            xc_tmp=(condition_x+1)/2.0*255.0
            #fre_xnoisy=DeFreq.depart_frequence(x0_tmp[0],t)
            fre_xsr=DeFreq.depart_frequence(xc_tmp[0],t)

            high_xsr=torch.from_numpy(fre_xsr[2].astype(float))
            mid_xsr=torch.from_numpy(fre_xsr[1].astype(float))
            low_xsr=torch.from_numpy(fre_xsr[0].astype(float))
            
            high_xsr=high_xsr / 255.0
            high_xsr =high_xsr * 2 - 1#标准化归一化
            mid_xsr=mid_xsr / 255.0
            mid_xsr =mid_xsr * 2 - 1#标准化归一化
            low_xsr=low_xsr / 255.0
            low_xsr =low_xsr * 2 - 1#标准化归一化
            '''
            high_xsr=fre_xsr[1]#256,256
            #print(high_xsr)
            #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/high_{}_image_noise.png'.format(t),high_xsr)

            similar_region_hf=DeFreq.calculate_ssim(fre_xnoisy[1],fre_xsr[1],t)#获取SR和HRnoise的高频区域的相似区域(256,256)(0,1)
            similar_region_hf= torch.from_numpy(similar_region_hf)



            gray_image1 =x0_tmp[0].cpu().numpy() #3,256,256
            gray_image1 = np.transpose(gray_image1, (1, 2, 0))#256,256,3

            # 转换为灰度图
            gray_xnoise = cv2.cvtColor(gray_image1, cv2.COLOR_BGR2GRAY)#(256, 256)

            gray_image2 = xc_tmp[0].cpu().numpy() 
            gray_image2 = np.transpose(gray_image2, (1, 2, 0))
            # 转换为灰度图
            gray_xsr = cv2.cvtColor(gray_image2, cv2.COLOR_BGR2GRAY)#(256, 256)
            gray_low=gray_xsr

            gray_xnoise_similar=torch.from_numpy(gray_xnoise*np.array(1-similar_region_hf))
            gray_xsr_similar=torch.from_numpy(gray_xsr*np.array(1-similar_region_hf))
            #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_gray_xsr.png'.format(t), gray_xsr)
            gray_xnoise=torch.from_numpy(gray_xnoise)
            gray_xsr=torch.from_numpy(gray_xsr)
            
            gray_low=gray_xsr

            gray_xsr_similar=gray_xsr_similar / 255.0
            gray_xsr_similar =gray_xsr_similar * 2 - 1#标准化归一化
            gray_low=gray_low / 255.0
            gray_low =gray_low * 2 - 1
            gray_xsr=gray_xsr / 255.0
            gray_xsr =gray_xsr * 2 - 1
            gray_xnoise=gray_xnoise / 255.0
            gray_xnoise =gray_xnoise * 2 - 1
            gray_xnoise_similar=gray_xnoise_similar/255.0
            gray_xnoise_similar=gray_xnoise_similar*2-1
            '''


            #depart_gray = torch.stack([gray_xnoise_similar,gray_xsr_similar,gray_low], dim=0).unsqueeze(0)
            depart_gray = torch.stack([low_xsr,mid_xsr,high_xsr], dim=0).unsqueeze(0)
            ###############
            #print("*************")
            if xh is not None:
                #print("^^^^^^^^^^^")
                continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                    np.random.uniform(
                        self.sqrt_alphas_cumprod_prev[t],
                        self.sqrt_alphas_cumprod_prev[t+1],
                        size=1
                    )
                ).to(condition_x.device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                    1, -1)
                noise=None
                noise = default(noise, lambda: torch.randn_like(condition_x))
                #x_save=np.transpose(x0_tmp[0].cpu().numpy(), (1, 2, 0))
                #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_xsave.png'.format(t), x_save)
                x_0_noisy = self.q_sample(#求加噪后的xsr
                    x_start=xh, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
                xh_tmp=(xh+1)/2.0*255.0
                fre_xh=DeFreq.depart_frequence(xh_tmp[0],t)
                high_xh=torch.from_numpy(fre_xh[2].astype(float))
                #print(high_xh.max())

                x_0_noisy[0][0][high_xh>torch.mean(high_xh)]=x[0][0][high_xh>torch.mean(high_xh)]
                x_0_noisy[0][1][high_xh>torch.mean(high_xh)]=x[0][1][high_xh>torch.mean(high_xh)]
                x_0_noisy[0][2][high_xh>torch.mean(high_xh)]=x[0][2][high_xh>torch.mean(high_xh)]

                #print(high_xh)

                x=x_0_noisy
                #####################################################
                
                
                #x_0_noisy_save=x[0].cpu().numpy()
                #print(x_sr_noisy_save.shape)
                #x_0_noisy_save=np.transpose(x_0_noisy_save, (1, 2, 0))
                #x_0_noisy_save=(x_0_noisy_save+1)/2.0*255.0
                #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_noise_xsr.png'.format(t), x_0_noisy_save)
            if xh is not None:
                #x_input=torch.cat((xh,x),1)
                x_input=torch.cat((condition_x,x),1)
            else:
                x_input=torch.cat((condition_x,x),1)
            
                ###############
            depart_gray=torch.cat((depart_gray.float(),depart_gray.float()),dim=1)
            #x_input=torch.cat((depart_gray.to('cuda'),condition_x, x),1)


            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x_input, noise_level))#UNet预测噪音4,12,256,256

        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None,xh=None):
        model_mean, model_log_variance = self.p_mean_variance(#得到噪音
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x,xh=xh)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            #这里的img替换为x_sr加入2000步噪声后的噪声图
            
            
            #############
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[699],
                    self.sqrt_alphas_cumprod_prev[700],
                    size=1
                )
            ).to(x.device)
            
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                1, -1)
            noise=None
            noise = default(noise, lambda: torch.randn_like(x))

            #############
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x,xh=None)#求xt-1
                #x_save=np.transpose(img[0].cpu().numpy(), (1, 2, 0))
                #x_save=(x_save+1)/2.0*255.0
                #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_xsave.png'.format(i), x_save)               
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            
            '''
            xh=img
            xsr=x[0].cpu().numpy()
            xsr=np.transpose(xsr, (1, 2, 0))
            xsr=(xsr+1)/2.0*255.0
            xhr=xh[0].cpu().numpy()
            xhr=np.transpose(xhr, (1, 2, 0))
            xhr=(xhr+1)/2.0*255.0
            enhance_img=Enhimg.enhance_add_image(xsr,xhr)
            cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/00_enhance.png', enhance_img)
            enhance_img=enhance_img / 255.0
            enhance_img =enhance_img * 2 - 1
            enhance_img=np.transpose(enhance_img, (2, 0, 1))
            enhance_img=torch.from_numpy(enhance_img).unsqueeze(0).to(torch.float)

            x_sr_noisy = self.q_sample(#求加噪后的xsr
                x_start=enhance_img.to('cuda'), continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
            
            img=x_sr_noisy
            for i in tqdm(reversed(range(0, 700)), desc='sampling loop time step', total=700):
                img = self.p_sample(img, i, condition_x=x,xh=enhance_img.to('cuda'))#求xt-1
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)                
            '''
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        #noise=none
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(#求加噪后的x0
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        
        #print(type(x_noisy))
        
        x_noisy_tmp=x_noisy
        x_sr_tmp=x_in['SR']
        x_noisy_tmp=(x_noisy_tmp+1)/2.0*255.0
        x_sr_tmp=(x_sr_tmp+1)/2.0*255.0
        for i in range(0,b):
            
            fre_xsr=DeFreq.depart_frequence(x_sr_tmp[i],t)
            high_xsr=torch.from_numpy(fre_xsr[2].astype(float))
            mid_xsr=torch.from_numpy(fre_xsr[1].astype(float))
            low_xsr=torch.from_numpy(fre_xsr[0].astype(float))
            
            high_xsr=high_xsr / 255.0
            high_xsr =high_xsr * 2 - 1#标准化归一化
            mid_xsr=mid_xsr / 255.0
            mid_xsr =mid_xsr * 2 - 1#标准化归一化
            low_xsr=low_xsr / 255.0
            low_xsr =low_xsr * 2 - 1#标准化归一化
            

            '''         
            fre_xnoisy=DeFreq.depart_frequence(x_noisy_tmp[i],t+0.5)
            #low_xnoisy= torch.from_numpy(fre_xnoisy[0])
            #low_xsr= torch.from_numpy(fre_xsr[0])


            #similar_region_lf=DeFreq.calculate_ssim(fre_xnoisy[0],fre_xsr[0])#获取SR和HRnoise的低频区域的相似区域(256,256)(0,1)
            #similar_region_lf= torch.from_numpy(similar_region_lf)
            similar_region_hf=DeFreq.calculate_ssim(fre_xnoisy[1],fre_xsr[1],t)#获取SR和HRnoise的高频区域的相似区域(256,256)(0,1)
            similar_region_hf= torch.from_numpy(similar_region_hf)
            gray_image1 = x_noisy_tmp[i].cpu().numpy() 
            
            
            gray_image1 = np.transpose(gray_image1, (1, 2, 0))
            
            # 转换为灰度图
            gray_xnoise = cv2.cvtColor(gray_image1, cv2.COLOR_BGR2GRAY)#(256, 256)

            gray_image2 = x_sr_tmp[i].cpu().numpy() 
            gray_image2 = np.transpose(gray_image2, (1, 2, 0))
            # 转换为灰度图
            gray_xsr = cv2.cvtColor(gray_image2, cv2.COLOR_BGR2GRAY)#(256, 256)
            gray_low=gray_xsr

            gray_xnoise_similar=torch.from_numpy(gray_xnoise*np.array(1-similar_region_hf))
            gray_xsr_similar=torch.from_numpy(gray_xsr*np.array(1-similar_region_hf))
            #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_gray_xsr.png'.format(t), gray_xsr)
            if t<=500:
                gray_xsr[similar_region_hf==0]=gray_xnoise[similar_region_hf==0]

            #cv2.imwrite('/home/haida/data/zuochenjuan/SR3_plus/save/{}_gray_xsr_enhance.png'.format(t), gray_xsr)
            gray_xnoise=torch.from_numpy(gray_xnoise)
            gray_xsr=torch.from_numpy(gray_xsr)
            gray_low=gray_xsr

            gray_xsr_similar=gray_xsr_similar / 255.0
            gray_xsr_similar =gray_xsr_similar * 2 - 1#标准化归一化
            gray_low=gray_low / 255.0
            gray_low =gray_low * 2 - 1
            gray_xsr=gray_xsr / 255.0
            gray_xsr =gray_xsr * 2 - 1
            gray_xnoise=gray_xnoise / 255.0
            gray_xnoise =gray_xnoise * 2 - 1
            gray_xnoise_similar=gray_xnoise_similar/255.0
            gray_xnoise_similar=gray_xnoise_similar*2-1
            '''

            
            depart_gray = torch.stack([low_xsr,mid_xsr,high_xsr], dim=0).unsqueeze(0)
            


            if i >= 1:
                x_in_depart=torch.cat((x_in_depart,depart_gray),0)
            else:
                x_in_depart=depart_gray
        

        if not self.conditional:
            print("x_input.shape")
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)#UNet求噪声
        else:
            #print(x_in['SR'].shape)
            #print(x_noisy.shape)
            #x_input1=torch.cat((x_sr_similar.to('cuda'),x_noisy_similar.to('cuda')),1)
            #x_input2=torch.cat((x_in['SR'], x_noisy),1)
            #print(x_input1.shape)
            #print(x_input2.shape)
            x_in_depart=x_in_depart.float()
            x_in_depart=torch.cat((x_in_depart,x_in_depart),dim=1)
            #x_input=torch.cat((x_in_depart.to('cuda'),x_in['SR'], x_noisy),1)#4,12,256,256
            x_input=torch.cat((x_in['SR'], x_noisy),1)#4,12,256,256

            
            x_recon = self.denoise_fn(x_input, continuous_sqrt_alpha_cumprod)#x=torch.cat([x_in['SR'], x_noisy], dim=1)4,12,256,256////2,12,256,256
            
            
            #print("$$$$$$$$$$$$$$$$$$$$$$$")
            #print(x_in['SR'])

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
