from functools import cached_property
import os
import json

from pydantic import BaseModel

from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


class VoiceConfig(BaseModel):
    name: str
    prompt_text: str
    format: str
    file: str

class Voice(BaseModel):
    name: str
    prompt_text: str
    prompt_latents: bytes

class VoiceMap(BaseModel):
    voices: list[Voice]

    @cached_property
    def voices_map(self):
        return {v.name: v for v in self.voices}

    def get(self, voice: str):
        return self.voices_map.get(voice)

async def load_voices(server: AsyncVoxCPMServerPool, voices_dir: str):
    with open(os.path.join(voices_dir, "config.json")) as f:
        configs = json.load(f)
    
    valid_configs = [VoiceConfig.model_validate(c) for c in configs]

    voices = []

    for c in valid_configs:
        with open(os.path.join(voices_dir, c.file), "rb") as f:
            prompt_latents = await server.encode_latents(f.read(), c.format)
        
        voices.append(Voice(name=c.name, prompt_text=c.prompt_text, prompt_latents=prompt_latents))
    
    return VoiceMap(voices=voices)
