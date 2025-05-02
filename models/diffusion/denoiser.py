from typing import List, Dict, Literal


"""
You use a denoiser to generate image from complete random noise using a trained diffuser model
"""


class DDPMDenoiser:
    """
    Simplest denoiser, it only subtracts the predicted noise from the noisy image until it gets the t_0 image
    """

    def __init__(self, num_step: int) -> None:
        pass
