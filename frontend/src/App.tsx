import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Shield, 
  Eye, 
  Brain, 
  Zap, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Info,
  Sun,
  Moon,
  ScanFace,
  ArrowRight,
  FileImage,
  Sparkles,
  Cpu,
  Network
} from 'lucide-react';

// Types
interface DetectionClass {
  id: string;
  name: string;
  label: string;
  description: string;
  icon: 'real' | 'deepfakes' | 'face2face' | 'faceswap' | 'neuraltextures';
  color: string;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const detectionClasses: DetectionClass[] = [
  {
    id: 'real',
    name: 'REAL',
    label: 'Authentique',
    description: 'Visage authentique, non manipulé',
    icon: 'real',
    color: 'text-success-400 bg-success-400/10 border-success-400/30',
  },
  {
    id: 'deepfakes',
    name: 'DEEPFAKES',
    label: 'Deepfakes',
    description: 'Synthèse faciale complète par auto-encodeur',
    icon: 'deepfakes',
    color: 'text-danger-500 bg-danger-500/10 border-danger-500/30',
  },
  {
    id: 'face2face',
    name: 'FACE2FACE',
    label: 'Face2Face',
    description: 'Transfert d\'expressions d\'un visage à un autre',
    icon: 'face2face',
    color: 'text-orange-400 bg-orange-400/10 border-orange-400/30',
  },
  {
    id: 'faceswap',
    name: 'FACESWAP',
    label: 'FaceSwap',
    description: 'Remplacement complet du visage',
    icon: 'faceswap',
    color: 'text-purple-400 bg-purple-400/10 border-purple-400/30',
  },
  {
    id: 'neuraltextures',
    name: 'NEURALTEXTURES',
    label: 'NeuralTextures',
    description: 'Modification subtile des textures de peau',
    icon: 'neuraltextures',
    color: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  },
];

// Animated Background Component
const AnimatedBackground = ({ isDark }: { isDark: boolean }) => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    {/* Grid Pattern */}
    <div 
      className={`absolute inset-0 bg-grid-pattern bg-grid opacity-50 ${isDark ? 'dark' : ''}`}
      style={{ backgroundSize: '50px 50px' }}
    />
    
    {/* Radial Glow */}
    <div className="absolute inset-0 bg-radial-glow" />
    
    {/* Animated Orbs */}
    <motion.div
      animate={{
        x: [0, 100, 0],
        y: [0, -50, 0],
        scale: [1, 1.2, 1],
      }}
      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      className={`absolute top-1/4 left-1/4 w-96 h-96 rounded-full blur-3xl ${
        isDark ? 'bg-accent-500/10' : 'bg-accent-400/20'
      }`}
    />
    <motion.div
      animate={{
        x: [0, -80, 0],
        y: [0, 80, 0],
        scale: [1, 1.3, 1],
      }}
      transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      className={`absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full blur-3xl ${
        isDark ? 'bg-purple-500/10' : 'bg-purple-400/15'
      }`}
    />
    
    
  </div>
);

// Class Card Component
const ClassCard = ({ 
  detectionClass, 
  isActive, 
  onClick,
  isDark = true
}: { 
  detectionClass: DetectionClass; 
  isActive: boolean; 
  onClick: () => void;
  isDark?: boolean;
}) => {
  const getIcon = () => {
    switch (detectionClass.icon) {
      case 'real':
        return <CheckCircle2 className="w-5 h-5" />;
      case 'deepfakes':
        return <XCircle className="w-5 h-5" />;
      case 'face2face':
        return <ArrowRight className="w-5 h-5" />;
      case 'faceswap':
        return <ScanFace className="w-5 h-5" />;
      case 'neuraltextures':
        return <Sparkles className="w-5 h-5" />;
    }
  };

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
      className={`
        relative p-4 rounded-2xl border-2 transition-all duration-300 text-left
        ${isActive 
          ? `${detectionClass.color} shadow-lg shadow-accent-500/20` 
          : 'border-white/10 bg-white/5 hover:border-white/20'
        }
        ${detectionClass.id === 'real' && isActive 
          ? 'border-success-400/50 bg-success-400/10' 
          : ''
        }
      `}
    >
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${detectionClass.color.split(' ')[1]}`}>
          {getIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className={`font-display font-semibold text-sm ${
            isActive 
              ? detectionClass.color.split(' ')[0] 
              : isDark ? 'text-white/80' : 'text-gray-800'
          }`}>
            {detectionClass.name}
          </h3>
          <p className={`text-xs mt-1 line-clamp-2 ${isDark ? 'text-white/50' : 'text-gray-600'}`}>
            {detectionClass.description}
          </p>
        </div>
      </div>
      
      {isActive && (
        <motion.div
          layoutId="activeGlow"
          className="absolute inset-0 rounded-2xl bg-accent-500/5"
          initial={false}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      )}
    </motion.button>
  );
};

// Upload Zone Component
const UploadZone = ({ 
  onUpload, 
  isDark 
}: { 
  onUpload: (file: File) => void;
  isDark: boolean;
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onUpload(file);
    }
  }, [onUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  }, [onUpload]);

  return (
    <motion.div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      animate={{
        borderColor: isDragging ? 'rgba(6, 182, 212, 0.5)' : 'rgba(255, 255, 255, 0.1)',
        backgroundColor: isDragging ? 'rgba(6, 182, 212, 0.05)' : 'rgba(255, 255, 255, 0.02)',
      }}
      className={`
        relative border-2 border-dashed rounded-3xl p-12 text-center
        transition-all duration-300 cursor-pointer
        ${isDark 
          ? 'hover:border-accent-500/30 hover:bg-accent-500/5' 
          : 'hover:border-accent-500/50 hover:bg-accent-50'
        }
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      
      <motion.div
        animate={{ scale: isDragging ? 1.1 : 1 }}
        className={`
          w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center
          ${isDark 
            ? 'bg-gradient-to-br from-accent-500/20 to-purple-500/20' 
            : 'bg-gradient-to-br from-accent-100 to-purple-100'
          }
        `}
      >
        <Upload className={`w-10 h-10 ${isDark ? 'text-accent-400' : 'text-accent-600'}`} />
      </motion.div>
      
      <h3 className={`text-xl font-display font-semibold mb-2 ${
        isDark ? 'text-white' : 'text-gray-900'
      }`}>
        Déposez votre image
      </h3>
      <p className={`text-sm mb-4 ${
        isDark ? 'text-white/50' : 'text-gray-500'
      }`}>
        ou cliquez pour sélectionner un fichier
      </p>
      <div className={`flex items-center justify-center gap-2 text-xs ${
        isDark ? 'text-white/30' : 'text-gray-400'
      }`}>
        <FileImage className="w-4 h-4" />
        <span>PNG, JPG, WEBP jusqu'à 10MB</span>
      </div>
    </motion.div>
  );
};

// Result Display Component
const ResultDisplay = ({ 
  result, 
  isDark 
}: { 
  result: { 
    class: string; 
    confidence: number; 
    heatmap?: string;
    heatmapOnly?: string;
    probabilities?: Record<string, number>;
    isFake?: boolean;
    inferenceTime?: number;
  } | null;
  isDark: boolean;
}) => {
  if (!result) return null;

  const detectedClass = detectionClasses.find(c => c.id === result.class);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        rounded-3xl p-6 border
        ${isDark 
          ? 'bg-dark-800/50 border-white/10' 
          : 'bg-white border-gray-200'
        }
      `}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-display font-semibold ${
          isDark ? 'text-white' : 'text-gray-900'
        }`}>
          Résultat de l'analyse
        </h3>
        <div className={`
          px-4 py-2 rounded-full text-sm font-semibold
          ${detectedClass?.color.split(' ')[1] || 'bg-gray-500/10'}
        `}>
          {detectedClass?.label || result.class}
        </div>
      </div>
      
      {/* Confidence Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-2">
          <span className={isDark ? 'text-white/60' : 'text-gray-600'}>
            Confiance
          </span>
          <span className={`font-mono font-semibold ${
            result.confidence > 80 ? 'text-success-400' : 
            result.confidence > 50 ? 'text-orange-400' : 'text-danger-400'
          }`}>
            {result.confidence.toFixed(1)}%
          </span>
        </div>
        <div className={`
          h-3 rounded-full overflow-hidden
          ${isDark ? 'bg-dark-600' : 'bg-gray-200'}
        `}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${result.confidence}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={`
              h-full rounded-full
              ${result.confidence > 80 ? 'bg-success-400' : 
                result.confidence > 50 ? 'bg-orange-400' : 'bg-danger-400'
              }
            `}
          />
        </div>
      </div>
      
      {/* Grad-CAM Heatmap Placeholder */}
      {result.heatmap && (
        <div className="mt-4">
          <p className="text-sm text-white/50 mb-2">Visualisation Grad-CAM</p>
          <div className="grid grid-cols-2 gap-3">
            <img src={`data:image/png;base64,${result.heatmapOnly}`}
                alt="heatmap" className="rounded-xl w-full" />
            <img src={`data:image/png;base64,${result.heatmap}`}
                alt="overlay" className="rounded-xl w-full" />
          </div>
        </div>
      )}
      {result.probabilities && (
        <div className="mt-4 space-y-2">
          {Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([cls, prob]) => (
              <div key={cls} className="flex items-center gap-3">
               <span className={`text-xs w-28 ${isDark ? 'text-white/40' : 'text-gray-500'}`}>{cls}</span>

                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-accent-400 rounded-full"
                      style={{ width: `${prob}%` }} />
                </div>
                <span className={`text-xs w-12 text-right ${isDark ? 'text-white/60' : 'text-gray-700'}`}>{prob}%</span>
              </div>
            ))}
        </div>
      )}
    </motion.div>
  );
};

// Main App Component
function App() {
  const [isDark, setIsDark] = useState(true);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{
    class: string;
    confidence: number;
    heatmap?: string;
    heatmapOnly?: string;
    probabilities?: Record<string, number>;
    isFake?: boolean;
    inferenceTime?: number;
  } | null>(null);

  const handleUpload = useCallback(async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);
    setResult(null);
    setIsAnalyzing(true);

    try {
      // Appel /gradcam pour avoir prédiction + heatmap en même temps
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_URL}/gradcam`, { method: "POST", body: fd });
      const data = await res.json();

      setResult({
        class: data.prediction,
        confidence: data.confidence,
        heatmap: data.images.overlay,        // base64 overlay
        heatmapOnly: data.images.heatmap,    // base64 heatmap seule
        probabilities: data.probabilities,
        isFake: data.is_fake,
        inferenceTime: data.inference_time_ms
      });
    } catch (e) {
      console.error("API Error:", e);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const toggleTheme = () => setIsDark(!isDark);

  return (
    <div className={`min-h-screen transition-colors duration-500 ${
      isDark 
        ? 'bg-dark-950 text-white' 
        : 'bg-light-50 text-gray-900'
    }`}>
      <AnimatedBackground isDark={isDark} />
      
      {/* Header */}
      <header className="relative z-10 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-3"
          >
            <div className={`
              w-12 h-12 rounded-xl flex items-center justify-center
              bg-gradient-to-br from-accent-500 to-purple-500
              shadow-lg shadow-accent-500/25
            `}>
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-display font-bold text-xl tracking-tight">
                DEEP<span className="text-accent-400">GUARD</span>
              </h1>
              <p className={`text-xs tracking-wider ${
                isDark ? 'text-white/40' : 'text-gray-500'
              }`}>
                AI DETECTION SYSTEM
              </p>
            </div>
          </motion.div>
          
          <motion.button
            onClick={toggleTheme}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`
              p-3 rounded-xl transition-all duration-300
              ${isDark 
                ? 'bg-dark-700 hover:bg-dark-600 text-white/70 hover:text-white' 
                : 'bg-white hover:bg-gray-100 text-gray-600 hover:text-gray-900'
              }
              shadow-lg
            `}
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </motion.button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className={`
            inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm mb-6
            ${isDark 
              ? 'bg-accent-500/10 text-accent-400 border border-accent-500/20' 
              : 'bg-accent-100 text-accent-700 border border-accent-200'
            }
          `}>
            <Brain className="w-4 h-4" />
            <span>Intelligence Artificielle Avancée</span>
          </div>
          
          <h2 className="font-display text-4xl md:text-6xl font-bold mb-6 leading-tight">
            Détection{' '}
            <span className="bg-gradient-to-r from-accent-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Intelligent
            </span>
            <br />
            des Deepfakes
          </h2>
          
          <p className={`text-lg max-w-2xl mx-auto ${
            isDark ? 'text-white/60' : 'text-gray-600'
          }`}>
            Analysez les images faciales et déterminez si elles sont authentiques 
            ou manipulées par IA. Localisez précisément les zones manipulées.
          </p>
        </motion.div>

        <div className="flex flex-col items-center">
          {/* Left Panel - Upload & Result */}
          <div className="w-full max-w-2xl space-y-6">
            <AnimatePresence mode="wait">
              {!uploadedImage ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <UploadZone onUpload={handleUpload} isDark={isDark} />
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className={`
                    rounded-3xl overflow-hidden border
                    ${isDark 
                      ? 'bg-dark-800/50 border-white/10' 
                      : 'bg-white border-gray-200'
                    }
                  `}
                >
                  <div className="relative w-full">
                    <img 
                      src={uploadedImage} 
                      alt="Uploaded" 
                      className="w-full h-auto object-contain max-h-96"
                    />
                    {isAnalyzing && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute inset-0 bg-dark-950/80 flex items-center justify-center"
                      >
                        <div className="text-center">
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            className="w-16 h-16 mx-auto mb-4"
                          >
                            <ScanFace className="w-16 h-16 text-accent-400" />
                          </motion.div>
                          <p className="text-accent-400 font-display font-semibold">
                            Analyse en cours...
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </div>
                  <div className="p-6">
                    <ResultDisplay result={result} isDark={isDark} />
                    <button
                      onClick={() => {
                        setUploadedImage(null);
                        setResult(null);
                      }}
                      className={`
                        mt-4 w-full py-3 rounded-xl font-medium transition-colors
                        ${isDark 
                          ? 'bg-dark-700 hover:bg-dark-600 text-white/70' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                        }
                      `}
                    >
                      Analyser une autre image
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Features Grid */}
            <div className="grid md:grid-cols-3 gap-4 w-full max-w-2xl mx-auto">
              {[
                { icon: Cpu, title: 'Analyse Rapide', desc: 'Résultats en secondes', color: 'accent' },
                { icon: Eye, title: 'Grad-CAM', desc: 'Localisation précise', color: 'purple' },
                { icon: Network, title: '5 Classes', desc: 'Détection multi-type', color: 'pink' },
              ].map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`
                    p-5 rounded-2xl border
                    ${isDark 
                      ? 'bg-dark-800/30 border-white/5 hover:border-white/10' 
                      : 'bg-white border-gray-200 hover:border-gray-300'
                    }
                    transition-all duration-300
                  `}
                >
                  <feature.icon className={`w-8 h-8 text-${feature.color}-400 mb-3`} />
                  <h3 className={`font-display font-semibold mb-1 ${
                    isDark ? 'text-white' : 'text-gray-900'
                  }`}>
                    {feature.title}
                  </h3>
                  <p className={`text-sm ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
                    {feature.desc}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>

        </div>

        {/* Types de manipulation - Bottom */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`
            mt-8 rounded-3xl p-6 border
            ${isDark 
              ? 'bg-dark-800/30 border-white/5' 
              : 'bg-white border-gray-200'
            }
          `}
        >
          <div className="flex items-center gap-2 mb-6">
            <Zap className={`w-5 h-5 ${isDark ? 'text-accent-400' : 'text-gray-400'}`} />
            <h3 className={`font-display font-semibold ${
              isDark ? 'text-white' : 'text-gray-900'
            }`}>
              Types de manipulation
            </h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {detectionClasses.map((detectionClass, index) => (
              <motion.div
                key={detectionClass.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <ClassCard
                  detectionClass={detectionClass}
                  isActive={selectedClass === detectionClass.id}
                  onClick={() => setSelectedClass(
                    selectedClass === detectionClass.id ? null : detectionClass.id
                  )}
                  isDark={isDark}
                />
                {selectedClass === detectionClass.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className={`
                      mt-2 p-4 rounded-xl text-sm
                      ${isDark ? 'bg-dark-700/50' : 'bg-gray-50'}
                    `}
                  >
                    <p className={isDark ? 'text-white/60' : 'text-transparent'}>
                      {detectionClass.description}
                    </p>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Info Box */}
          <div className={`
            mt-6 p-4 rounded-xl border
            ${isDark 
              ? 'bg-accent-500/5 border-accent-500/20' 
              : 'bg-gray-50 border-gray-200'
            }
          `}>
            <div className="flex gap-3">
              <Info className={`w-5 h-5 flex-shrink-0 mt-0.5 ${isDark ? 'text-accent-400' : 'text-gray-400'}`} />
              <p className={`text-sm ${isDark ? 'text-white/50' : 'text-gray-600'}`}>
                Ce système utilise des réseaux de neurones profonds entraînés 
                sur des milliers d'échantillons pour une précision optimale.
              </p>
            </div>
          </div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className={`
            flex flex-col md:flex-row items-center justify-between gap-4
            ${isDark ? 'text-white/30' : 'text-gray-500'}
          `}>
            <p className="text-sm">
              © 2026 DeepGuard. Système de détection de deepfakes.
            </p>
            <div className="flex items-center gap-6 text-sm">
              <span className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-success-400" />
                Précision 99%+
              </span>
              <span className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-accent-400" />
                Sécurité Avancée
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
