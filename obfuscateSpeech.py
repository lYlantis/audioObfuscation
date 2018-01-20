import numpy as np
from scipy import signal
import soundfile as sf
import talkboxlpc3 as tlk
from matplotlib.mlab import find

vowelFrames = None

def VUS(x, L, fs):
    '''
    vocalic syllable detector
    '''
    pmin = int(fs/500) #500 is highest average human pitch (child)
    #pmax = int(fs/80)  #80 is lowest average human pitch (male)
    pitch = 0

    xw = x*np.hamming(L)

    #Clipping function
    maxVal = np.amax(xw, axis=0)
    CL = 0.3*maxVal

    xwClipped = xw

    xwClipped = np.where(xwClipped >= CL, 1,
             (np.where(xwClipped<=-CL, -1, 0)))

    #Pitch Detector
    Rn = np.zeros(shape=(L,1))

    Rn = freq_from_autocorr(xwClipped, fs)

    Rmax = np.amax(Rn, axis=0)

    p_idx = np.argmax(Rn, axis =0)

    R0 = np.sum(np.abs(xwClipped))

    #zero crossings
    ZC = 0.5*(np.sum(np.abs(np.sign(x[1:-1])-np.sign(x[:-2]))))/L

    #VUS discrimination
    rms = np.sqrt(np.mean(np.square(x)))
    if rms > 0.04:
        if Rmax > 0.2*R0:
            v = 1
            pitch = fs/(p_idx + pmin)
        else:
            v=0.67
    elif ZC > 0.35:
        v = 0.33
    else:
        v = 0

    return pitch, v

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def prepareReplacementCoefficients():
    global vowelFrames
    vowel1, fs2 = sf.read('uw.wav')
    vowel2, fs3 = sf.read('eh.wav')
    order = 20
    fs = 16000
    L = round(30*fs/1000) #30 ms frames
    
    # LPC of UW and EH
    vowelFrames = int(np.floor(len(vowel1)/L)-1)
    vowel1LPCs = np.zeros(shape=(vowelFrames,order+1))
    vowel2LPCs = np.zeros(shape=(vowelFrames,order+1))
    
    for i in range(0,vowelFrames):
        vowel1frame = vowel1[i*L:i*L+L]*np.hamming(L)
        vowel2frame = vowel2[i*L:i*L+L]*np.hamming(L)
        a_vowel1 = tlk.lpc_ref(vowel1frame,order)
        a_vowel2 = tlk.lpc_ref(vowel2frame,order)
        vowel1LPCs[i,:] = a_vowel1
        vowel2LPCs[i,:] = a_vowel2

    return vowel1LPCs, vowel2LPCs

def obfuscate(frames, replacementVowels):

    inp = frames
    fs = 16000
     
    order = 20
    L = round(30*fs/1000) #30 ms frames
    R = round(15*fs/1000) #15 ms overlap
    n = 0
    
    syllable = 0
    syllableCount = 0
    vowelSwitch = 0
    Mframe = 0
    Nframe =0
    
    result = np.zeros(shape=(len(inp)+L,1))
    voiced = np.zeros(shape=(len(inp)+L,1))
    sout = np.zeros(shape=(L,1))
    pitches = []
    
    count = 1
    x = np.zeros(shape=(len(inp[n:n+L]),1))
    
    while(n+L <= len(inp)):
        #-- s[n] --
        x = inp[n:n+L]
        xw = x*np.hamming(L)
    
        #-- LPC of s[n] --
        a = tlk.lpc_ref(xw, order)
        _, g, _ = tlk.levinson_1d(xw,order)
        a = a[np.newaxis]
        a.T
        
        a = np.real(a)
        if g<0:
            g = g*-1
        g = np.sqrt(g)
        g = np.array(g, ndmin=1, copy=False)
        b = a.copy()
        b = np.multiply(b, -1)
        b[0][0] = 0
        xp = signal.lfilter(b[0],1,xw)
        en = xw-xp

        # voiced, unvoiced, silence (VUS) discriminator
        pitch, v = VUS(x,L,fs)
        pitches.append(pitch)
    
        #Look ahead 1 frame
        if (n+R+L <= len(inp)):
            x_future = inp[n+R:n+R+L]
            pitch_future, v_future = VUS(x_future,L,fs)
    
        if v == .33 and v_future>.6:
            v = .67
    
        # -- Choosing UW LPCs --
        if v == 1:
            Mframe += 1
        elif v>.6 and syllable == 0:
            syllableCount += 1
            vowelSwitch = np.mod(syllableCount, 2)
            syllable = 1
        elif v == 0 and syllable == 1:
            syllable = 0
            Mframe = 0
        
        if Mframe == 0:
            sout = signal.lfilter(g, a[0], en)
        elif Mframe < vowelFrames:
            if vowelSwitch == 0:
                sout = signal.lfilter(g, replacementVowels[0][Mframe,:],en)
            else:
                sout = signal.lfilter(g, replacementVowels[1][Mframe,:],en)
        else:
            if vowelSwitch == 0:
                sout = signal.lfilter(g, replacementVowels[0][-1,:],en)
            else:
                sout = signal.lfilter(g, replacementVowels[1][-1,:],en)
                
        #Maintain the same amplitude as original frame
        sout = sout*(np.amax(x, axis=0)/np.amax(np.abs(sout), axis=0))
        
        for i in range(L):
            if n == 0:
                result[i] = sout[i]
                voiced[i] = v
            else:
                result[n+i] = result[n+i] + sout[i]
                voiced[n+i] = v

        Nframe += 1
        n += R
        count += 1
    
    pitches = np.array(pitches)
    result = result/np.nanmax(np.abs(result), axis=0)
    
    averagePitch = np.mean(np.where(pitches>0))

    #Avg pitch male = 110hzm st. dev = 25hz
    if averagePitch >85 and averagePitch < 135:
        maleOrFemale = 'male'
    else:
        maleOrFemale = 'female'

    return frames, averagePitch, maleOrFemale
 
def getAudio():
    '''
    Records audio for X seconds, based on parameters in heading
    Parameters
    ----------
    Nothing
    
    Returns
    -------
    frames : numpy array with audio data
    '''
    CHUNK = 512  # Changing the recording seconds might require a change to this value
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5  # Use .125 in real setting. 5 seconds was used for testing audio validity
    CHUNKS_TO_RECORD = int(RATE / CHUNK * RECORD_SECONDS)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    logging.debug("* Recording audio...")

    frames = np.empty((CHUNKS_TO_RECORD * CHUNK), dtype="int16")
    logging.info('Begin Recording...')
    time.sleep(1)
    logging.info('Now')
    for i in range(0, CHUNKS_TO_RECORD):
        audioString = stream.read(CHUNK)
        frames[i * CHUNK:(i + 1) * CHUNK] = np.fromstring(audioString, dtype="int16")

    logging.debug("* done recording\n")
    
    logging.debug("closing stream")
    
    #frames = frames / (2.0 ** 15)  # convert it to floating point between -1 and 1
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames

if __name__ == "__main__":
    frames = getAudio()
    vowel1, vowel2 = prepareReplacementCoefficients()
    a, b, c, = obfuscate(frames,[vowel1,vowel2])
    sd.play(a, 16000)
