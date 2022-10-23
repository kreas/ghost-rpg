import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import uuid
import io
import boto3

HUGGING_FACE = "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"
R2_KEY = "89dd139487431efac2995b76c15182ac"
R2_SECRET = "a6e95920d09a8b6b915fd671b18899de5e85ee117874eb8f84f603746f3bf430"
R2_ENDPOINT = "https://5291ed0146b6509eecceb89f2d915f95.r2.cloudflarestorage.com"
BUCKET = 'scrollrack-dev'

device = 'cuda'

lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_type=torch.float16,
    # revision="fp16",
    use_auth_token=HUGGING_FACE
).to(device)

pipe.enable_attention_slicing()

def dummy(images, **kwargs):
    return images, False

def send(image, file_prefix):
    filename = f"{file_prefix}_{str(uuid.uuid4())}.jpg"
    mem_file = io.BytesIO()
    image.save(mem_file, format='JPEG')
    mem_file.seek(0)

    s3 = boto3.client(
        service_name='s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
    )

    resp = s3.upload_fileobj(
        mem_file,
        Bucket=BUCKET,
        Key=filename,
        ExtraArgs={
            'ContentType': 'image/jpeg',
        }
    )

pipe.safety_checker = dummy

def create_latents(width, height, seed):
    generator = torch.Generator(device=device)
    new_seed = generator.seed() if seed == 0 or seed is None else seed
    generator = generator.manual_seed(new_seed)

    image_latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8,  width // 8),
        device=device,
        generator=generator
    )

    return new_seed, image_latents

style = "detailed portrait, cell shaded, 4 k, concept art, by wlop, ilya kuvshinov, artgerm, krenz cushart, greg rutkowski, pixiv. cinematic dramatic atmosphere, sharp focus, volumetric lighting, cinematic lighting, studio quality"
prompt = "beautiful female ninja, oni mask, wearing cyberpunk intricate streetwear, clevage"
file_prefix="cyborg_ohni_mask"
height = 640
width = 512
seed = 0
steps = 50
guidance_scale = 7.5

for n in range(300):
    with autocast("cuda"):
        for prompt in prompts:
            full_prompt = f"{prompt['prompt']}, {style}"
            
            image_seed, latents = create_latents(width, height, seed)
            image = pipe(full_prompt, guidance_scale=guidance_scale, width=width,
                        height=height, num_inference_steps=steps, latents=latents)["sample"][0]
            send(image, prompt.get("prefix"))