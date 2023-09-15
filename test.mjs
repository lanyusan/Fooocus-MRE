const base_models = [
  "sd_xl_base_1.0_0.9vae.safetensors",
  "sdxlNijiSpecial_sdxlNijiSE.safetensors",
];

const count = 1;

for (let i = 0; i < count; i++) {
  for (const modelFile of base_models) {
    const resp = await fetch("http://localhost:8000/test", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        base_model_name: modelFile,
        prompt:
          " a (((black))) chinese dragon, ((stars and lighting)) in background",
      }),
    });
    const json = await resp.json();
    console.log(JSON.stringify(json, null, 2));
  }
}
