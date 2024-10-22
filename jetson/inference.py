import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.OUT)
# Load TensorRT engine and allocate buffers
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
def nptype_fix(trt_type):
    import numpy as np
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,  # Use np.bool_ instead of np.bool
    }
    return mapping[trt_type]

# Then modify your allocate_buffers function:
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = nptype_fix(engine.get_binding_dtype(binding))  # Use the new function here
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return the output data.
    return [out['host'] for out in outputs]


# Load the TensorRT engine
engine_path = r'/home/gourav/Desktop/vss/TensorRT-optimized-driver-drowsiness-system/jetson/ssd_mobilenet_fp16.plan'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, engine_path)

# Create an execution context
context = engine.create_execution_context()

# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Assuming the input shape is (1, height, width, channels)
input_shape = engine.get_binding_shape(0)
batch_size, height, width, channels = input_shape

img_size = 224
face_cascade = cv2.CascadeClassifier(r'/home/gourav/Desktop/vss/TensorRT-optimized-driver-drowsiness-system/jetson/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error")
        break

    eyes_cascade = cv2.CascadeClassifier(r'/home/gourav/Desktop/vss/TensorRT-optimized-driver-drowsiness-system/jetson/haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyes_cascade.detectMultiScale(gray, 1.1, 6)
    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        eyess = eyes_cascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
            continue
        else:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]

    final_image = cv2.resize(eyes_roi, (img_size, img_size))
    final_image = final_image.astype(np.float32) / 255.0  # Rescale pixel values
    # Add batch dimension (1, height, width, channels)
    final_image = np.expand_dims(final_image, axis=0)

    # Prepare input data
    inputs[0]['host'] = np.ascontiguousarray(final_image)

    output = do_inference(context, bindings=bindings,inputs=inputs, outputs=outputs, stream=stream)[0]


    # Process the output
    predicted_class = (output > 3).astype(int).flatten()

    if (predicted_class == 1):    
        print("driver drowsiness system online")
    else:
        counter = counter+1
        if counter > 10:
            print("Driver is drowsy")
            GPIO.output(7, GPIO.True)
            time.sleep(3)
            GPIO.output(7, GPIO.False)
            counter = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()