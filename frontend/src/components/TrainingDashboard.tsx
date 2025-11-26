/**
 * Training Dashboard Component
 * Displays real-time RL training progress with Genesis simulation visualization
 */

import React, { useEffect, useState, useRef } from 'react';
import './TrainingDashboard.css';

interface TrainingDashboardProps {
  /** WebSocket connection for training data */
  wsUrl?: string;
  /** Number of environments to display in grid */
  envGridSize?: number;
  /** Show detailed metrics */
  showDetailedMetrics?: boolean;
  /** Backend URL for API calls */
  backendUrl?: string;
}

interface TrainingMetrics {
  iteration: number;
  total_steps: number;
  mean_reward: number;
  std_reward: number;
  mean_episode_length: number;
  std_episode_length: number;
  learning_rate: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
  best_reward: number;
  best_env_idx: number;
  fps: number;
  time_elapsed: number;
}

interface EnvironmentState {
  env_idx: number;
  reward: number;
  episode_length: number;
  done: boolean;
  thumbnail?: string; // base64 image
}

const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  wsUrl = 'ws://localhost:8000/ws/genesis/training',
  envGridSize = 16,
  showDetailedMetrics = true,
  backendUrl = 'http://localhost:8000'
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const chartCanvasRef = useRef<HTMLCanvasElement>(null);

  const [connected, setConnected] = useState(false);
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [environments, setEnvironments] = useState<EnvironmentState[]>([]);
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<number | null>(null);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('🎓 Training dashboard connected');
      setConnected(true);

      // Request initial state
      ws.send(JSON.stringify({ type: 'get_training_status' }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'training_metrics') {
          setMetrics(message.data);

          // Update reward history
          setRewardHistory(prev => {
            const newHistory = [...prev, message.data.mean_reward];
            // Keep last 100 data points
            return newHistory.slice(-100);
          });
        } else if (message.type === 'environment_states') {
          setEnvironments(message.data);
        } else if (message.type === 'training_status') {
          setTraining(message.data.training);
        } else if (message.type === 'environment_thumbnail') {
          // Update specific environment thumbnail
          const { env_idx, image } = message.data;
          setEnvironments(prev =>
            prev.map(env =>
              env.env_idx === env_idx
                ? { ...env, thumbnail: image }
                : env
            )
          );
        }
      } catch (error) {
        console.error('Training dashboard message error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('Training dashboard disconnected');
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [wsUrl]);

  // Draw reward chart
  useEffect(() => {
    if (!chartCanvasRef.current || rewardHistory.length === 0) return;

    const canvas = chartCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Calculate min/max for scaling
    const minReward = Math.min(...rewardHistory);
    const maxReward = Math.max(...rewardHistory);
    const range = maxReward - minReward || 1;

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw reward line
    ctx.strokeStyle = '#4a90e2';
    ctx.lineWidth = 2;
    ctx.beginPath();

    rewardHistory.forEach((reward, i) => {
      const x = (i / (rewardHistory.length - 1)) * width;
      const y = height - ((reward - minReward) / range) * height;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw current value
    if (rewardHistory.length > 0) {
      const lastReward = rewardHistory[rewardHistory.length - 1];
      const y = height - ((lastReward - minReward) / range) * height;

      ctx.fillStyle = '#4a90e2';
      ctx.beginPath();
      ctx.arc(width - 5, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [rewardHistory]);

  const startTraining = async () => {
    try {
      await fetch(`${backendUrl}/api/genesis/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm: 'ppo',
          num_envs: envGridSize * envGridSize,
          max_iterations: 10000
        })
      });
      setTraining(true);
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const stopTraining = async () => {
    try {
      await fetch(`${backendUrl}/api/genesis/training/stop`, {
        method: 'POST'
      });
      setTraining(false);
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const viewEnvironment = (envIdx: number) => {
    setSelectedEnv(envIdx);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'focus_environment',
        env_idx: envIdx
      }));
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${mins}m ${secs}s`;
  };

  return (
    <div className="training-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h2>🎓 RL Training Dashboard</h2>
          <div className={`status-badge ${training ? 'training' : 'idle'}`}>
            {training ? '🔄 Training' : '⏸️ Idle'}
          </div>
        </div>
        <div className="header-right">
          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Left Panel - Metrics */}
        <div className="metrics-panel">
          {/* Key Metrics */}
          <div className="metrics-section">
            <h3>📊 Key Metrics</h3>
            <div className="metric-cards">
              <div className="metric-card">
                <div className="metric-label">Mean Reward</div>
                <div className="metric-value">
                  {metrics?.mean_reward.toFixed(2) || '0.00'}
                </div>
                <div className="metric-subtext">
                  ±{metrics?.std_reward.toFixed(2) || '0.00'}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Best Reward</div>
                <div className="metric-value highlight">
                  {metrics?.best_reward.toFixed(2) || '0.00'}
                </div>
                <div className="metric-subtext">
                  Env #{metrics?.best_env_idx || 0}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Episode Length</div>
                <div className="metric-value">
                  {metrics?.mean_episode_length.toFixed(0) || '0'}
                </div>
                <div className="metric-subtext">
                  ±{metrics?.std_episode_length.toFixed(0) || '0'}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Training FPS</div>
                <div className="metric-value">
                  {metrics?.fps.toFixed(0) || '0'}
                </div>
                <div className="metric-subtext">
                  {metrics ? `${(metrics.fps * (envGridSize * envGridSize)).toFixed(0)} steps/s` : '0 steps/s'}
                </div>
              </div>
            </div>
          </div>

          {/* Reward Chart */}
          <div className="metrics-section">
            <h3>📈 Reward History</h3>
            <canvas
              ref={chartCanvasRef}
              width={400}
              height={200}
              className="reward-chart"
            />
          </div>

          {/* Detailed Metrics */}
          {showDetailedMetrics && metrics && (
            <div className="metrics-section">
              <h3>🔬 Training Details</h3>
              <div className="detail-list">
                <div className="detail-item">
                  <span>Iteration:</span>
                  <span>{metrics.iteration}</span>
                </div>
                <div className="detail-item">
                  <span>Total Steps:</span>
                  <span>{metrics.total_steps.toLocaleString()}</span>
                </div>
                <div className="detail-item">
                  <span>Learning Rate:</span>
                  <span>{metrics.learning_rate.toExponential(2)}</span>
                </div>
                <div className="detail-item">
                  <span>Policy Loss:</span>
                  <span>{metrics.policy_loss.toFixed(4)}</span>
                </div>
                <div className="detail-item">
                  <span>Value Loss:</span>
                  <span>{metrics.value_loss.toFixed(4)}</span>
                </div>
                <div className="detail-item">
                  <span>Entropy:</span>
                  <span>{metrics.entropy.toFixed(4)}</span>
                </div>
                <div className="detail-item">
                  <span>Time Elapsed:</span>
                  <span>{formatTime(metrics.time_elapsed)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="metrics-section">
            <h3>🎮 Controls</h3>
            <div className="control-buttons">
              {!training ? (
                <button
                  onClick={startTraining}
                  className="btn btn-primary"
                  disabled={!connected}
                >
                  ▶️ Start Training
                </button>
              ) : (
                <button
                  onClick={stopTraining}
                  className="btn btn-danger"
                >
                  ⏹️ Stop Training
                </button>
              )}
              <button
                className="btn btn-secondary"
                disabled={!connected}
                onClick={() => {
                  if (wsRef.current) {
                    wsRef.current.send(JSON.stringify({ type: 'save_checkpoint' }));
                  }
                }}
              >
                💾 Save Checkpoint
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Environment Grid */}
        <div className="environments-panel">
          <h3>
            🌍 Parallel Environments
            <span className="env-count">{envGridSize * envGridSize} envs</span>
          </h3>

          <div
            className="env-grid"
            style={{
              gridTemplateColumns: `repeat(${envGridSize}, 1fr)`
            }}
          >
            {Array.from({ length: envGridSize * envGridSize }).map((_, idx) => {
              const env = environments.find(e => e.env_idx === idx);
              const isBest = idx === metrics?.best_env_idx;
              const isSelected = idx === selectedEnv;

              return (
                <div
                  key={idx}
                  className={`env-cell ${isBest ? 'best' : ''} ${isSelected ? 'selected' : ''}`}
                  onClick={() => viewEnvironment(idx)}
                >
                  {env?.thumbnail ? (
                    <img
                      src={`data:image/jpeg;base64,${env.thumbnail}`}
                      alt={`Environment ${idx}`}
                      className="env-thumbnail"
                    />
                  ) : (
                    <div className="env-placeholder">
                      <span className="env-idx">{idx}</span>
                    </div>
                  )}

                  {isBest && <div className="best-badge">👑</div>}

                  {env && (
                    <div className="env-overlay">
                      <div className="env-reward">
                        R: {env.reward.toFixed(1)}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingDashboard;
