import argparse
import sys
import numpy as np
import cv2
import os

from perceptual.filterbank import Steerable
from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter

# determine what OpenCV version we are using
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False

def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq, accel='cpu'):
    
    use_cuda = (accel == 'cuda')
    if use_cuda:
        import cupy as cp
        xp = cp
        print("[CUDA] GPU Acceleration Enabled")
    else:
        xp = np
        print("[CPU] Standard Computation Enabled")

    # initialize the steerable complex pyramid
    steer = Steerable(5, use_cuda=use_cuda)
    pyArr = Pyramid2arr(steer)

    print("Reading:", vidFname, end=" ")

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    if USE_CV2:
        # OpenCV 2.x interface
        vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
        func_fourcc = cv.CV_FOURCC
    else:
        # OpenCV 3.x interface
        vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv2.CAP_PROP_FPS))
        func_fourcc = cv2.VideoWriter_fourcc

    if np.isnan(fps):
        fps = 30
    import time
    import platform
    import cpuinfo
    from tqdm import tqdm

    # Gather System Info
    cpu_info = cpuinfo.get_cpu_info()
    os_info = platform.system() + " " + platform.release()
    py_ver = platform.python_version()
    
    # Calculate Memory (approximate available RAM via psutil if available, fallback otherwise)
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024.**3), 1)
        ram_str = f"{ram_gb} GB"
    except ImportError:
        ram_str = "Unknown (psutil not installed)"

    print("="*60)
    print("  Hardware Information")
    print("="*60)
    print(f"  OS           : {os_info}")
    print(f"  Python       : {py_ver}")
    print(f"  Accel Mode   : {accel.upper()}")
    
    if use_cuda:
        try:
            # Try getting GPU info from CuPy or nvidia-smi
            gpu_name = xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            cuda_ver = xp.cuda.runtime.runtimeGetVersion()
            # approximate memory
            free_mem, total_mem = xp.cuda.runtime.memGetInfo()
            gpu_mem = round(total_mem / (1024.**3), 1)
            print(f"  GPU          : {gpu_name}")
            print(f"  GPU Memory   : {gpu_mem} GB")
            print(f"  CUDA Version : {cuda_ver}")
        except Exception:
            print("  GPU          : Unknown CUDA Device")
    else:
        num_threads = args.threads if 'args' in globals() and hasattr(args, 'threads') and args.threads > 0 else 1
        print(f"  CPU          : {cpu_info.get('brand_raw', 'Unknown')}")
        print(f"  CPU Cores    : {cpu_info.get('count', 'Unknown')} logical")
        print(f"  RAM          : {ram_str}")
        print(f"  Threads Used : {num_threads}")
    print("="*60)

    print("\n" + "="*60)
    print("  Input Video Information")
    print("="*60)
    print(f"  File         : {os.path.basename(vidFname)}")
    print(f"  Resolution   : {width} x {height}")
    print(f"  FPS          : {fps:.2f}")
    
    # video Writer
    fourcc = func_fourcc(*'MJPG')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    
    # how many frames
    nrFrames = min(vidFrames, maxFrames)
    duration = nrFrames / fps if fps > 0 else 0
    print(f"  Frames       : {nrFrames}")
    print(f"  Duration     : {duration:.2f} s")
    print("="*60 + "\n")

    print("Starting EVM processing...")
    start_time = time.time()

    # setup temporal filter
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fpsForBandPass, outfun=lambda x: x[0])
    #filter = ButterBandpassFilter(1, lowFreq, highFreq, fps=fpsForBandPass)

    from concurrent.futures import ThreadPoolExecutor
    import threading
    import queue

    num_threads = args.threads if 'args' in globals() and hasattr(args, 'threads') and args.threads > 0 else 1

    if num_threads <= 1:
        # ---- Single Threaded Fallback ----
        pbar = tqdm(total=nrFrames+windowSize, desc=f"Video Processing ({accel.upper()})", unit="frames")
        for frameNr in range( nrFrames + windowSize ):
            pbar.update(1)
            if frameNr < nrFrames:
                ret, im = vidReader.read()
                if not ret or im is None:
                    break
                
                if len(im.shape) > 2:
                    grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                else:
                    grayIm = im
                    
                if use_cuda:
                    grayIm = xp.asarray(grayIm)

                coeff = steer.buildSCFpyr(grayIm)
                arr = pyArr.p2a(coeff)
                phases = xp.angle(arr)

                filter.update(xp.asarray([phases]))

                try:
                    filteredPhases = filter.next()
                except StopIteration:
                    continue

                magnifiedPhases = (phases - filteredPhases) + filteredPhases*factor
                newArr = xp.abs(arr) * xp.exp(magnifiedPhases * 1j)
                newCoeff = pyArr.a2p(newArr)
                out = steer.reconSCFpyr(newCoeff)

                out[out>255] = 255
                out[out<0] = 0
                
                if use_cuda:
                    out = xp.asnumpy(out)
                
                rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
                rgbIm[:,:,0] = out
                rgbIm[:,:,1] = out
                rgbIm[:,:,2] = out
                
                vidWriter.write(cv2.convertScaleAbs(rgbIm).astype(np.uint8))
        pbar.close()
    else:
        # ---- Multi-Threaded Pipeline ----
        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading
        
        mode_str = f"{accel.upper()} parallel" if accel == 'cpu' else f"{accel.upper()}"
        pbar_dec = tqdm(total=nrFrames, desc=f"Pyramids Generation ({mode_str})", position=0)
        pbar_fil = tqdm(total=nrFrames, desc=f"Pyramids Filtering", position=1)
        pbar_rec = tqdm(total=nrFrames, desc=f"Video Reconstruction ({mode_str})", position=2)
        
        # Queues
        q_raw = queue.Queue(maxsize=windowSize*2)            # (frameNr, grayIm)
        q_phases = queue.PriorityQueue()                     # (frameNr, phases, arr)
        q_filter_out = queue.Queue(maxsize=windowSize*2)     # (frameNr, magnifiedPhases, arr)
        q_reconstruct = queue.PriorityQueue()                # (frameNr, bgrIm)
        
        # Worker 1: Producer (Read + Spatial Decompose)
        def decomposer_worker():
            while True:
                item = q_raw.get()
                if item is None:
                    q_raw.task_done()
                    break
                f_nr, gray_img = item
                if use_cuda:
                    gray_img = xp.asarray(gray_img)
                    
                coeff = steer.buildSCFpyr(gray_img)
                arr = pyArr.p2a(coeff)
                phases = xp.angle(arr)
                q_phases.put((f_nr, phases, arr))
                pbar_dec.update(1)
                q_raw.task_done()

        # Worker 2: Consumer (Reconstruct)
        def reconstruct_worker():
            while True:
                item = q_filter_out.get()
                if item is None:
                    q_filter_out.task_done()
                    break
                f_nr, mag_phases, arr = item
                newArr = xp.abs(arr) * xp.exp(mag_phases * 1j)
                newCoeff = pyArr.a2p(newArr)
                out = steer.reconSCFpyr(newCoeff)
                out[out>255] = 255
                out[out<0] = 0
                
                if use_cuda:
                    out = xp.asnumpy(out)
                
                rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
                rgbIm[:,:,0] = out
                rgbIm[:,:,1] = out
                rgbIm[:,:,2] = out
                
                res = cv2.convertScaleAbs(rgbIm).astype(np.uint8)
                q_reconstruct.put((f_nr, res))
                q_filter_out.task_done()
                
        # Main Thread components
        pool_dec = ThreadPoolExecutor(max_workers=num_threads)
        pool_rec = ThreadPoolExecutor(max_workers=num_threads)
        
        for _ in range(num_threads):
            pool_dec.submit(decomposer_worker)
            pool_rec.submit(reconstruct_worker)

        # Start a dedicated writer thread to clear reconstruct queue
        frames_to_write = nrFrames
        written_count = 0
        
        def writer_loop():
            nonlocal written_count
            next_write_f = 0
            buf = {}
            while written_count < frames_to_write:
                try:
                    # Give a timeout so we can exit cleanly if frames_to_write drops
                    item = q_reconstruct.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if item is None:
                    break
                    
                f_nr, res = item
                buf[f_nr] = res
                while next_write_f in buf:
                    vidWriter.write(buf.pop(next_write_f))
                    next_write_f += 1
                    written_count += 1
                    pbar_rec.update(1)
                q_reconstruct.task_done()
                
        writer_thr = threading.Thread(target=writer_loop)
        writer_thr.start()

        # Phase Sequence Execution
        import time
        read_idx = 0
        filter_idx = 0
        phases_buffer = {}

        while written_count < frames_to_write:
            made_progress = False
            
            # 1. Feed the Video Reader
            # Keep the decoder queue slightly fuller than the filter window
            while read_idx < nrFrames and not q_raw.full():
                ret, img = vidReader.read()
                if ret and img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)>2 else img
                    q_raw.put((read_idx, gray))
                    read_idx += 1
                    made_progress = True
                else:
                    frames_to_write = read_idx
                    break

            # 2. Sequential Filtering
            try:
                # Poll decoded frames non-blockingly
                while not q_phases.empty():
                    f, p, a = q_phases.get_nowait()
                    phases_buffer[f] = (p, a)
                    q_phases.task_done()
                    made_progress = True
            except queue.Empty:
                pass

            while filter_idx in phases_buffer:
                p, a = phases_buffer.pop(filter_idx)
                
                filter.update(xp.asarray([p]))
                try:
                    filt_p = filter.next()
                    mag_phases = (p - filt_p) + filt_p * factor
                    q_filter_out.put((filter_idx, mag_phases, a))
                except StopIteration:
                    # Filter is filling up its sliding window
                    frames_to_write -= 1 
                
                pbar_fil.update(1)
                filter_idx += 1
                made_progress = True

            if not made_progress:
                time.sleep(0.005)

        # Teardown
        for _ in range(num_threads):
            q_raw.put(None)
            q_filter_out.put(None)
            
        pool_dec.shutdown(wait=True)
        pool_rec.shutdown(wait=True)
        
        q_reconstruct.put(None)
        writer_thr.join()
        
        pbar_rec.n = nrFrames
        pbar_rec.refresh()
        pbar_dec.close()
        pbar_fil.close()
        pbar_rec.close()

    vidReader.release()
    vidWriter.release()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("  Processing Complete!")
    print(f"  Total Time   : {total_time:.2f} s ({total_time/60:.2f} min)")
    print("="*60 + "\n")
    print(f"Video saved to: {vidFnameOut}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase-Based Video Motion Magnification (EVM Python)")
    parser.add_argument('-v', '--video_path', type=str, required=True, help="Input video path")
    parser.add_argument('-s', '--saving_path', type=str, required=True, help="Output sequence path (.mp4, .avi)")
    parser.add_argument('-a', '--alpha', type=float, default=20.0, help="Amplification factor")
    parser.add_argument('-lo', '--low_omega', type=float, default=72.0, help="Min frequency (hz)")
    parser.add_argument('-ho', '--high_omega', type=float, default=92.0, help="Max frequency (hz)")
    parser.add_argument('-ws', '--window_size', type=int, default=30, help="Sliding window size")
    parser.add_argument('-mf', '--max_frames', type=int, default=60000, help="Max frames to process")
    parser.add_argument('-f', '--fps', type=int, default=600, help="FPS for bandpass. -1 for video native FPS.")
    parser.add_argument('-t', '--threads', type=int, default=1, help="Number of CPU workers for multithreading")
    parser.add_argument('-acc', '--accel', choices=['cpu', 'cuda'], default='cpu', help="Acceleration backend: cpu or cuda")
    
    args = parser.parse_args()
    
    # Ensure parents directories of saving_path exists
    os.makedirs(os.path.dirname(os.path.abspath(args.saving_path)), exist_ok=True)
    
    phaseBasedMagnify(
        vidFname=args.video_path,
        vidFnameOut=args.saving_path,
        maxFrames=args.max_frames,
        windowSize=args.window_size,
        factor=args.alpha,
        fpsForBandPass=args.fps,
        lowFreq=args.low_omega,
        highFreq=args.high_omega,
        accel=args.accel
    )
