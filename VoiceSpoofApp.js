import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Animated,
  Platform,
  Dimensions,
  Pressable,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

// ─────────────────────────────────────────
//  CONFIG — replace with your machine's IP
// ─────────────────────────────────────────
import { BACKEND_URL } from './config';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─────────────────────────────────────────
//  THEME (mirrors the web UI colour palette)
// ─────────────────────────────────────────
const COLORS = {
  bg1: '#0a0a1a',
  bg2: '#0f0c29',
  card: 'rgba(255,255,255,0.055)',
  border: 'rgba(255,255,255,0.10)',
  accent: '#7c3aed',
  accent2: '#4f46e5',
  real: '#22c55e',
  spoof: '#ef4444',
  warn: '#f59e0b',
  text: '#e2e8f0',
  muted: '#94a3b8',
  purple: '#c4b5fd',
};

// ─────────────────────────────────────────
//  WAVE VISUALIZER (animated bars)
// ─────────────────────────────────────────
function WaveVisualizer({ isRecording }) {
  const bars = 30;
  const anims = useRef(Array.from({ length: bars }, () => new Animated.Value(0.15))).current;
  const loopRef = useRef(null);

  useEffect(() => {
    if (isRecording) {
      const animations = anims.map((anim, i) => {
        return Animated.loop(
          Animated.sequence([
            Animated.delay(i * 40),
            Animated.timing(anim, {
              toValue: 0.2 + Math.random() * 0.8,
              duration: 300 + Math.random() * 400,
              useNativeDriver: true,
            }),
            Animated.timing(anim, {
              toValue: 0.1 + Math.random() * 0.3,
              duration: 200 + Math.random() * 300,
              useNativeDriver: true,
            }),
          ])
        );
      });
      loopRef.current = Animated.parallel(animations);
      loopRef.current.start();
    } else {
      if (loopRef.current) loopRef.current.stop();
      anims.forEach(a => {
        Animated.spring(a, {
          toValue: 0.15,
          useNativeDriver: true,
          friction: 8,
        }).start();
      });
    }
    return () => {
      if (loopRef.current) loopRef.current.stop();
    };
  }, [isRecording]);

  return (
    <View style={styles.waveContainer}>
      <View style={styles.waveInner}>
        {anims.map((anim, i) => (
          <Animated.View
            key={i}
            style={[
              styles.waveBar,
              {
                transform: [{ scaleY: anim }],
                backgroundColor: isRecording
                  ? `rgba(0, ${180 + Math.floor(i * 2.5)}, 255, 0.8)`
                  : 'rgba(0, 200, 255, 0.2)',
              },
            ]}
          />
        ))}
      </View>
      <Text style={[styles.waveLabel, isRecording && styles.waveLabelActive]}>
        {isRecording ? '● LIVE WAVEFORM' : '● LIVE WAVEFORM'}
      </Text>
    </View>
  );
}

// ─────────────────────────────────────────
//  WINDOW BAR CHART
// ─────────────────────────────────────────
function WindowBars({ windowScores, threshold }) {
  if (!windowScores || windowScores.length <= 1) return null;
  return (
    <View style={styles.windowBarsWrap}>
      <Text style={styles.windowBarsTitle}>Per-window scores</Text>
      <View style={styles.windowBarsRow}>
        {windowScores.map((s, i) => {
          const isReal = s > threshold;
          const barH = Math.max(6, s * 46);
          return (
            <View key={i} style={styles.windowBarCol}>
              <View
                style={[
                  styles.windowBarFill,
                  {
                    height: barH,
                    backgroundColor: isReal ? '#4ade80' : '#f87171',
                  },
                ]}
              />
              <Text style={styles.windowBarLabel}>{s.toFixed(2)}</Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

// ─────────────────────────────────────────
//  RESULT CARD
// ─────────────────────────────────────────
function ResultCard({ result }) {
  if (!result) return null;

  const barAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(barAnim, {
      toValue: result.error ? 0 : (result.confidence || 0) / 100,
      duration: 700,
      useNativeDriver: false,
    }).start();
  }, [result]);

  if (result.error) {
    return (
      <View style={styles.resultCard}>
        <Text style={[styles.resultLabel, { color: COLORS.warn, fontSize: 20 }]}>
          ⚠️ Error
        </Text>
        <Text style={styles.confidenceText}>{result.error}</Text>
      </View>
    );
  }

  let labelText = '';
  let labelColor = COLORS.spoof;
  let barColor = ['#b91c1c', '#f87171'];

  if (result.result === 'REAL VOICE') {
    labelText = '✅ REAL VOICE';
    labelColor = COLORS.real;
    barColor = ['#16a34a', '#4ade80'];
  } else if (result.result && (result.result.includes('% HUMAN') || result.result.startsWith('MIXED'))) {
    labelText = '🟡 ' + result.result;
    labelColor = '#fbbf24';
    barColor = ['#d97706', '#fbbf24'];
  } else {
    labelText = '🚨 AI VOICE';
    labelColor = COLORS.spoof;
    barColor = ['#b91c1c', '#f87171'];
  }

  return (
    <View style={styles.resultCard}>
      <Text style={[styles.resultLabel, { color: labelColor }]}>{labelText}</Text>

      {/* Progress bar */}
      <View style={styles.progressBarWrap}>
        <Animated.View
          style={[
            styles.progressBarFill,
            {
              width: barAnim.interpolate({
                inputRange: [0, 1],
                outputRange: ['0%', '100%'],
              }),
              backgroundColor: barColor[1],
            },
          ]}
        />
      </View>

      <Text style={styles.confidenceText}>Confidence: {result.confidence}%</Text>

      {result.raw_score !== undefined && (
        <Text style={styles.rawScore}>
          Final score: {result.raw_score}
          {result.threshold !== undefined ? ` | threshold: ${result.threshold}` : ''}
        </Text>
      )}

      {result.cnn_score !== undefined && (
        <Text style={styles.rawScore}>
          CNN: {result.cnn_score} − penalty: {result.penalty ?? 0} = {result.raw_score}
          {'\n'}Real votes: {result.real_votes ?? '?'}/{result.windows_used ?? '?'} windows
        </Text>
      )}

      <WindowBars windowScores={result.window_scores} threshold={result.threshold ?? 0.5} />
    </View>
  );
}

// ─────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────
export default function VoiceSpoofApp() {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [statusMsg, setStatusMsg] = useState('Press Start to begin recording');
  const [micAudioUri, setMicAudioUri] = useState(null);
  const [fileAudioUri, setFileAudioUri] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState('');

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  // Save section
  const [showSaveSection, setShowSaveSection] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');
  const [saveMsgColor, setSaveMsgColor] = useState(COLORS.muted);
  const [realBtnText, setRealBtnText] = useState('💾 Save as Real Voice');
  const [spoofBtnText, setSpoofBtnText] = useState('🚨 Save as Spoof Voice');
  const [isSaving, setIsSaving] = useState(false);

  // Sound objects
  const recordingRef = useRef(null);
  const timerRef = useRef(null);
  const lastWavUri = useRef(null);
  const [micSound, setMicSound] = useState(null);
  const [fileSound, setFileSound] = useState(null);
  const [micPlaying, setMicPlaying] = useState(false);
  const [filePlaying, setFilePlaying] = useState(false);

  // Pulse animation
  const pulseAnim = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    if (isRecording) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 0.3, duration: 500, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      if (recordingRef.current) recordingRef.current.stopAndUnloadAsync().catch(() => {});
      if (micSound) micSound.unloadAsync().catch(() => {});
      if (fileSound) fileSound.unloadAsync().catch(() => {});
    };
  }, []);

  // ── Format timer ──────────────────────────────────────
  const formatTime = (secs) => {
    const mm = String(Math.floor(secs / 60)).padStart(2, '0');
    const ss = String(secs % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  };

  // ── Send WAV to server ────────────────────────────────
  const sendToServer = async (uri, filename) => {
    setIsAnalyzing(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
        name: filename,
        type: 'audio/wav',
      });

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({ error: 'Network error: ' + err.message + '\n\nMake sure backend is running and IP is correct in config.js' });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── START RECORDING ───────────────────────────────────
  const startRecording = async () => {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        Alert.alert('Permission Denied', 'Microphone access is required to record audio.');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync({
        android: {
          extension: '.wav',
          outputFormat: Audio.AndroidOutputFormat.DEFAULT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
          sampleRate: 22050,
          numberOfChannels: 1,
          bitRate: 128000,
        },
        ios: {
          extension: '.wav',
          audioQuality: Audio.IOSAudioQuality.HIGH,
          sampleRate: 22050,
          numberOfChannels: 1,
          bitRate: 128000,
          linearPCMBitDepth: 16,
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {},
      });

      await rec.startAsync();
      recordingRef.current = rec;

      setIsRecording(true);
      setRecordingSeconds(0);
      setStatusMsg('Recording... Speak clearly');
      setMicAudioUri(null);
      setShowSaveSection(false);
      setSaveMsg('');
      setRealBtnText('💾 Save as Real Voice');
      setSpoofBtnText('🚨 Save as Spoof Voice');
      lastWavUri.current = null;

      let secs = 0;
      timerRef.current = setInterval(() => {
        secs++;
        setRecordingSeconds(secs);
      }, 1000);

    } catch (err) {
      Alert.alert('Recording Error', err.message);
    }
  };

  // ── STOP RECORDING & ANALYZE ──────────────────────────
  const stopRecording = async () => {
    if (!recordingRef.current) return;
    if (recordingSeconds < 2) {
      Alert.alert('Too Short', 'Please record for at least 2 seconds.');
      return;
    }

    clearInterval(timerRef.current);
    setIsRecording(false);
    setStatusMsg('Analyzing...');

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });

      setMicAudioUri(uri);
      lastWavUri.current = uri;

      setStatusMsg('Sending to server...');
      await sendToServer(uri, 'recording.wav');

      setShowSaveSection(true);
      setStatusMsg('Press Start to record again');
    } catch (err) {
      setStatusMsg('Error: ' + err.message);
      Alert.alert('Error', err.message);
    }
  };

  // ── PLAY / PAUSE MIC AUDIO ────────────────────────────
  const toggleMicPlayback = async () => {
    try {
      if (micSound) {
        if (micPlaying) {
          await micSound.pauseAsync();
          setMicPlaying(false);
        } else {
          await micSound.playFromPositionAsync(0);
          setMicPlaying(true);
        }
        return;
      }
      if (!micAudioUri) return;
      const { sound } = await Audio.Sound.createAsync({ uri: micAudioUri }, { shouldPlay: true });
      setMicSound(sound);
      setMicPlaying(true);
      sound.setOnPlaybackStatusUpdate(status => {
        if (status.didJustFinish) setMicPlaying(false);
      });
    } catch (err) {
      Alert.alert('Playback Error', err.message);
    }
  };

  // ── PICK FILE ────────────────────────────────────────
  const pickFile = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: ['audio/*'],
        copyToCacheDirectory: true,
      });

      if (res.canceled || !res.assets || res.assets.length === 0) return;

      const asset = res.assets[0];
      setFileAudioUri(asset.uri);
      setSelectedFileName(asset.name);
      setResult(null);
    } catch (err) {
      Alert.alert('File Error', err.message);
    }
  };

  // ── PLAY / PAUSE FILE AUDIO ────────────────────────────
  const toggleFilePlayback = async () => {
    try {
      if (fileSound) {
        if (filePlaying) {
          await fileSound.pauseAsync();
          setFilePlaying(false);
        } else {
          await fileSound.playFromPositionAsync(0);
          setFilePlaying(true);
        }
        return;
      }
      if (!fileAudioUri) return;
      const { sound } = await Audio.Sound.createAsync({ uri: fileAudioUri }, { shouldPlay: true });
      setFileSound(sound);
      setFilePlaying(true);
      sound.setOnPlaybackStatusUpdate(status => {
        if (status.didJustFinish) setFilePlaying(false);
      });
    } catch (err) {
      Alert.alert('Playback Error', err.message);
    }
  };

  // ── ANALYZE FILE ─────────────────────────────────────
  const analyzeFile = async () => {
    if (!fileAudioUri) {
      Alert.alert('No File', 'Please select an audio file first.');
      return;
    }
    await sendToServer(fileAudioUri, 'upload.wav');
  };

  // ── SAVE SAMPLE ──────────────────────────────────────
  const saveSample = async (type) => {
    if (!lastWavUri.current) {
      Alert.alert('No Recording', 'No recording found. Please record your voice first.');
      return;
    }

    setIsSaving(true);
    setSaveMsg('Saving...');
    setSaveMsgColor(COLORS.muted);

    const isReal = type === 'real';
    const endpoint = isReal ? '/save-real' : '/save-spoof';
    const filename = isReal ? 'real_sample.wav' : 'spoof_sample.wav';

    try {
      const formData = new FormData();
      formData.append('file', {
        uri: Platform.OS === 'android' ? lastWavUri.current : lastWavUri.current.replace('file://', ''),
        name: filename,
        type: 'audio/wav',
      });

      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = await response.json();

      if (data.error) {
        setSaveMsg('❌ Error: ' + data.error);
        setSaveMsgColor('#f87171');
      } else {
        setSaveMsg('✅ ' + data.message + ' · Retrain when you have 50+ new samples.');
        setSaveMsgColor('#4ade80');
        if (isReal) setRealBtnText('✅ Saved!');
        else setSpoofBtnText('✅ Saved!');
      }
    } catch (err) {
      setSaveMsg('❌ Network error: ' + err.message);
      setSaveMsgColor('#f87171');
    } finally {
      setIsSaving(false);
    }
  };

  // ─────────────────────────────────────────
  //  RENDER
  // ─────────────────────────────────────────
  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Header ── */}
        <View style={styles.hero}>
          <Text style={styles.heroTitle}>🛡️ Voice Spoof Detection</Text>
          <Text style={styles.heroSub}>Detect AI-generated or spoofed voices in real time</Text>
        </View>

        {/* ── Upload File Card ── */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>📂 Upload Audio File</Text>

          <TouchableOpacity style={styles.fileDrop} onPress={pickFile} activeOpacity={0.75}>
            <Ionicons name="cloud-upload-outline" size={32} color="rgba(167,139,250,0.6)" />
            <Text style={styles.dropLabel}>Tap to choose a file (WAV / MP3 / FLAC / M4A)</Text>
            {selectedFileName ? (
              <Text style={styles.fileName}>{selectedFileName}</Text>
            ) : null}
          </TouchableOpacity>

          {fileAudioUri && (
            <TouchableOpacity style={[styles.btn, styles.btnSave, { marginBottom: 10 }]} onPress={toggleFilePlayback}>
              <Ionicons name={filePlaying ? 'pause' : 'play'} size={16} color="#fff" />
              <Text style={styles.btnText}>{filePlaying ? 'Pause' : 'Play Selected File'}</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[styles.btn, styles.btnPrimary, styles.btnFull]}
            onPress={analyzeFile}
            disabled={isAnalyzing || !fileAudioUri}
            activeOpacity={0.8}
          >
            <Ionicons name="search" size={16} color="#fff" />
            <Text style={styles.btnText}>🔍 Analyze File</Text>
          </TouchableOpacity>
        </View>

        {/* ── Live Recording Card ── */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>🎙️ Live Microphone Recording</Text>

          <View style={styles.tipBox}>
            <Text style={styles.tipText}>
              ⚠️ Record at least <Text style={{ fontWeight: '700' }}>4 seconds</Text> of clear speech for the best accuracy (multi-window analysis).
            </Text>
          </View>

          {/* Timer */}
          {isRecording && (
            <Text style={styles.timerDisplay}>{formatTime(recordingSeconds)}</Text>
          )}

          {/* Status */}
          <View style={styles.statusRow}>
            {isRecording && (
              <Animated.View style={[styles.pulseDot, { opacity: pulseAnim }]} />
            )}
            <Text style={styles.statusMsg}>{statusMsg}</Text>
          </View>

          {/* Waveform */}
          <WaveVisualizer isRecording={isRecording} />

          {/* Record Buttons */}
          <View style={styles.recordRow}>
            <TouchableOpacity
              style={[styles.btn, styles.btnRecord, { flex: 1 }]}
              onPress={startRecording}
              disabled={isRecording || isAnalyzing}
              activeOpacity={0.8}
            >
              <Ionicons name="radio-button-on" size={16} color="#fff" />
              <Text style={styles.btnText}>⏺ Start Recording</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.btn, styles.btnStop, { flex: 1 }]}
              onPress={stopRecording}
              disabled={!isRecording}
              activeOpacity={0.8}
            >
              <Ionicons name="stop" size={16} color="#fff" />
              <Text style={styles.btnText}>⏹ Stop & Analyze</Text>
            </TouchableOpacity>
          </View>

          {/* Mic Playback */}
          {micAudioUri && !isRecording && (
            <TouchableOpacity
              style={[styles.btn, styles.btnSave, { marginTop: 10 }]}
              onPress={toggleMicPlayback}
            >
              <Ionicons name={micPlaying ? 'pause' : 'play'} size={16} color="#fff" />
              <Text style={styles.btnText}>{micPlaying ? 'Pause Recording' : 'Play Recording'}</Text>
            </TouchableOpacity>
          )}

          {/* Save Section */}
          {showSaveSection && (
            <View style={styles.saveSection}>
              <Text style={styles.saveTitle}>🎯 Help improve the model — save this recording:</Text>
              <View style={styles.saveRow}>
                <TouchableOpacity
                  style={[styles.btn, styles.btnSave, { flex: 1 }]}
                  onPress={() => saveSample('real')}
                  disabled={isSaving}
                  activeOpacity={0.8}
                >
                  <Text style={styles.btnText}>{realBtnText}</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.btn, styles.btnSaveSpoof, { flex: 1 }]}
                  onPress={() => saveSample('spoof')}
                  disabled={isSaving}
                  activeOpacity={0.8}
                >
                  <Text style={styles.btnText}>{spoofBtnText}</Text>
                </TouchableOpacity>
              </View>
              {saveMsg ? (
                <Text style={[styles.saveMsg, { color: saveMsgColor }]}>{saveMsg}</Text>
              ) : null}
            </View>
          )}
        </View>

        {/* ── Spinner ── */}
        {isAnalyzing && (
          <View style={styles.spinnerWrap}>
            <ActivityIndicator size="large" color="#a78bfa" />
            <Text style={styles.spinnerText}>Analyzing audio...</Text>
          </View>
        )}

        {/* ── Result Card ── */}
        {result && !isAnalyzing && <ResultCard result={result} />}

        {/* Bottom padding */}
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

// ─────────────────────────────────────────
//  STYLES
// ─────────────────────────────────────────
const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.bg1,
  },
  scroll: {
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 16,
    paddingBottom: 60,
    alignItems: 'center',
  },

  // Hero
  hero: {
    alignItems: 'center',
    marginBottom: 28,
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: '800',
    color: '#a78bfa',
    letterSpacing: -0.5,
    marginBottom: 6,
    textAlign: 'center',
  },
  heroSub: {
    fontSize: 13,
    color: COLORS.muted,
    textAlign: 'center',
  },

  // Card
  card: {
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 14,
    padding: 22,
    width: '100%',
    maxWidth: 560,
    marginBottom: 18,
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: COLORS.purple,
    marginBottom: 16,
    letterSpacing: 0.2,
  },

  // File Drop
  fileDrop: {
    borderWidth: 2,
    borderColor: 'rgba(167,139,250,0.35)',
    borderStyle: 'dashed',
    borderRadius: 10,
    padding: 22,
    alignItems: 'center',
    marginBottom: 12,
  },
  dropLabel: {
    color: COLORS.muted,
    fontSize: 13,
    marginTop: 8,
    textAlign: 'center',
  },
  fileName: {
    color: '#a78bfa',
    fontSize: 13,
    fontWeight: '600',
    marginTop: 6,
    textAlign: 'center',
  },

  // Tip
  tipBox: {
    backgroundColor: 'rgba(245,158,11,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(245,158,11,0.22)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 14,
  },
  tipText: {
    color: '#fbbf24',
    fontSize: 12,
  },

  // Timer
  timerDisplay: {
    fontSize: 36,
    fontWeight: '800',
    color: '#f87171',
    textAlign: 'center',
    letterSpacing: 4,
    marginBottom: 4,
  },

  // Status
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  pulseDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#f87171',
    marginRight: 6,
  },
  statusMsg: {
    color: COLORS.muted,
    fontSize: 13,
    textAlign: 'center',
  },

  // Waveform
  waveContainer: {
    width: '100%',
    height: 90,
    backgroundColor: '#000',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(0,200,255,0.15)',
    marginBottom: 14,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  waveInner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    paddingHorizontal: 8,
    gap: 2,
  },
  waveBar: {
    width: 3,
    height: 60,
    borderRadius: 2,
  },
  waveLabel: {
    position: 'absolute',
    top: 8,
    left: 10,
    fontSize: 9,
    fontWeight: '600',
    letterSpacing: 1.2,
    color: 'rgba(0,230,255,0.4)',
    textTransform: 'uppercase',
  },
  waveLabelActive: {
    color: 'rgba(0,230,255,0.9)',
  },

  // Buttons
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 9,
  },
  btnText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  btnFull: {
    width: '100%',
    marginTop: 4,
  },
  btnPrimary: {
    backgroundColor: '#7c3aed',
  },
  btnRecord: {
    backgroundColor: '#dc2626',
  },
  btnStop: {
    backgroundColor: '#16a34a',
  },
  btnSave: {
    backgroundColor: '#0891b2',
  },
  btnSaveSpoof: {
    backgroundColor: '#c2410c',
  },

  // Record row
  recordRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 4,
  },

  // Save section
  saveSection: {
    marginTop: 14,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: 14,
  },
  saveTitle: {
    fontSize: 12,
    color: COLORS.muted,
    marginBottom: 10,
  },
  saveRow: {
    flexDirection: 'row',
    gap: 10,
  },
  saveMsg: {
    fontSize: 12,
    marginTop: 8,
  },

  // Spinner
  spinnerWrap: {
    alignItems: 'center',
    padding: 20,
  },
  spinnerText: {
    color: COLORS.muted,
    fontSize: 13,
    marginTop: 8,
  },

  // Result card
  resultCard: {
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 14,
    padding: 24,
    width: '100%',
    maxWidth: 560,
    alignItems: 'center',
    marginBottom: 18,
  },
  resultLabel: {
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: 0.5,
    marginBottom: 8,
    textAlign: 'center',
  },
  progressBarWrap: {
    width: '100%',
    height: 11,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 100,
    overflow: 'hidden',
    marginVertical: 14,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 100,
  },
  confidenceText: {
    fontSize: 14,
    color: '#cbd5e1',
    marginBottom: 4,
    textAlign: 'center',
  },
  rawScore: {
    fontSize: 11,
    color: '#475569',
    marginTop: 4,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    textAlign: 'center',
  },

  // Window bars
  windowBarsWrap: {
    marginTop: 14,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: 14,
    width: '100%',
    alignItems: 'center',
  },
  windowBarsTitle: {
    fontSize: 10,
    color: '#64748b',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  windowBarsRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 56,
    gap: 4,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  windowBarCol: {
    alignItems: 'center',
    width: 28,
  },
  windowBarFill: {
    width: '100%',
    borderRadius: 3,
  },
  windowBarLabel: {
    fontSize: 8,
    color: '#64748b',
    marginTop: 2,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
});
