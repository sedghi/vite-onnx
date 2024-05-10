import { useEffect, useRef, useState } from "react"
import ort from "onnxruntime-web/webgpu"
import "./App.css"

function App() {
  const [session , setSession] = useState(null)
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current) return

    const load = async (url: string) => {
      console.debug("🚀 ~ url:", url);
      const bytes = await fetch(url).then((response) => response.arrayBuffer())
      
      const onnxSession = await ort.InferenceSession.create(bytes, {
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
      
      setSession(onnxSession)
      initialized.current = true
    }

    // load("http://localhost:5173/sam_h/vit_h_encoder.onnx")
    load("http://localhost:5173/sam_h/vit_h_decoder.onnx")
  }, [])

  if (!session) return <div>Loading...</div>
  
  return <div>
    Loaded! Input names:
    {session.handler.inputNames.map((name) => <div key={name}>{name}</div>)}</div>
}

export default App
