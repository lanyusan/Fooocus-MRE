let count = 2; //1000;

const base_models = [
  "sd_xl_base_1.0_0.9vae.safetensors",
  "sdvn7Nijistylexl_v1.safetensors",
];

for (let i = 0; i < count; i++)
  for (const modelFile of base_models) {
    await fetch("http://localhost:8000/runsync", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        input: {
          base_model_name: modelFile,
          prompt: "a puppy",
        },
      }),
    });
  }
