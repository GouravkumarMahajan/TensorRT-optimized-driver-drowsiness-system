import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import Jetson.GPIO as GPIO
import time

# Disable GPIO warnings
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)

def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def nptype_fix(trt_type):
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,  
    }
    return mapping[trt_type]

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = nptype_fix(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

def process_frame(frame, eyes_cascade, img_size):
    if frame is None:
        return None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyes_cascade.detectMultiScale(gray, 1.1, 6)
    
    eyes_roi = None
    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        eyess = eyes_cascade.detectMultiScale(roi_gray)
        
        if len(eyess) > 0:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
                if eyes_roi is not None and eyes_roi.size > 0:
                    try:
                        eyes_roi = cv2.resize(eyes_roi, (img_size, img_size))
                        return eyes_roi
                    except Exception as e:
                        print(f"Error resizing ROI: {e}")
                        continue
    return None

def main():
    # Load the TensorRT engine
    engine_path = r'/home/csed/gourav/TensorRT-optimized-driver-drowsiness-system/jetson/model_fp16.plan'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(trt_runtime, engine_path)

    # Create an execution context
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    input_shape = engine.get_binding_shape(0)
    batch_size, height, width, channels = input_shape

    img_size = 224
    face_cascade = cv2.CascadeClassifier(r'/home/csed/gourav/TensorRT-optimized-driver-drowsiness-system/jetson/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(r'/home/csed/gourav/TensorRT-optimized-driver-drowsiness-system/jetson/haarcascade_eye_tree_eyeglasses.xml')
    
    # Check if cascades loaded successfully
    if face_cascade.empty() or eyes_cascade.empty():
        print("Error: Could not load cascade classifiers")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    counter = -2
    no_eyes_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Process frame to get eyes ROI
            eyes_roi = process_frame(frame, eyes_cascade, img_size)
            
            if eyes_roi is None:
                no_eyes_counter += 1
                if no_eyes_counter > 30:  # About 1 second at 30 fps
                    print("No eyes detected!")
                    no_eyes_counter = 0
                continue

            # Reset no_eyes_counter when eyes are detected
            no_eyes_counter = 0
            
            # Prepare image for inference
            final_image = eyes_roi.astype(np.float32) / 255.0
            final_image = np.expand_dims(final_image, axis=0)
            inputs[0]['host'] = np.ascontiguousarray(final_image)

            # Run inference
            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]

            # Process the output
            predicted_class = (output > 3).astype(int).flatten()

            if predicted_class == 1:
                print("Driver drowsiness system online: Driver awake")
                counter = 0
            else:
                counter += 1
                print("Driver drowsiness system online: suspecting driver is drowsy")
                if counter > 0:
                    print("Driver drowsiness system online: Driver is drowsy")
                    GPIO.output(7, GPIO.HIGH)
                    time.sleep(3)
                    GPIO.output(7, GPIO.LOW)
                    counter = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping the program...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

