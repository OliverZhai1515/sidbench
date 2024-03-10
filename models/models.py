
MODELS = [
    {
        "model_name": "UnivFD",
        "trained_on": "progan",
        "ckpt": "./weights/univfd/fc_weights.pth"
    },
    {
        "model_name": "CNNDetect",
        "trained_on": "progan",
        "ckpt": "./weights/cnndetect/blur_jpg_prob0.5.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "latent_diffusion",
        "ckpt": "./weights/dimd/corvi22_latent_model.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "progan",
        "ckpt": "./weights/dimd/corvi22_progan_model.pth"
    },
    {
        "model_name": "DIMD",
        "trained_on": "stylegan2",
        "ckpt": "./weights/dimd/gandetection_resnet50nodown_stylegan2.pth"
    },
    {
        "model_name": "LGrad",
        "trained_on": "progan",
        "ckpt": "./weights/lgrad/LGrad.pth"
    },
    {
        "model_name": "FreqDetect",
        "trained_on": "progan",
        "ckpt": "./weights/freqdetect/DCTAnalysis.pth"
    },
    {
        "model_name": "Rine",
        "trained_on": "progan",
        "ckpt": "./weights/rine/model_1class_trainable.pth",
        "ncls": "1class"
    },
    {
        "model_name": "Rine",
        "trained_on": "latent_diffusion",
        "ckpt": "./weights/rine/model_ldm_trainable.pth",
        "ncls": "ldm"
    },
    {
        "model_name": "NPR",
        "trained_on": "progan",
        "ckpt": "./weights/npr/NPR.pth"
    },
    {
        "model_name": "RPTC",
        "trained_on": "progan",
        "ckpt": "./weights/rptc/RPTC.pth"
    },
    {
        "model_name": "Fusing",
        "trained_on": "progan",
        "ckpt": "./weights/fusing/PSM.pth"
    },
    {
        "model_name": "GramNet",
        "trained_on": "progan",
        "ckpt": "./weights/gramnet/Gram.pth"
    }
]
