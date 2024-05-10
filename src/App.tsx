import { useEffect, useRef } from "react"
import ort from "onnxruntime-web/webgpu"
import "./App.css"

function App() {
  const initialized = useRef(false)

  useEffect(() => {
    const load = async (url: string) => {
      console.debug("🚀 ~ url:", url);
      const bytes = await fetch(url).then((response) => response.arrayBuffer())
      
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["webgpu"],
        enableMemPattern: false,
        enableCpuMemArena: false,
        extra: {
          session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1",
          },
        },
        interOpNumThreads: 4,
        intraOpNumThreads: 2,
      })
      console.log(session)
      initialized.current = true
    }

    // load("http://localhost:5173/sam_h/vit_h_encoder.onnx")
    load("http://localhost:5173/sam_h/vit_h_decoder.onnx")
  }, [])

  return <div>hi</div>
}

export default App
