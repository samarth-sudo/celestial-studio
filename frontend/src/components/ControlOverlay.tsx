/**
 * Control Overlay Component
 *
 * Displays keyboard controls and recording UI for robot teleoperation
 * Shows different control schemes based on robot type
 */

import React, { useState } from 'react';
import './ControlOverlay.css';

interface ControlOverlayProps {
  robotType: 'mobile' | 'arm' | 'drone';
  isRecording: boolean;
  recordingSteps?: number;
  recordingTime?: number;
  isConnected: boolean;
  onStartRecording: (taskName: string) => void;
  onStopRecording: () => void;
  onReset?: () => void;
}

export function ControlOverlay({
  robotType,
  isRecording,
  recordingSteps = 0,
  recordingTime = 0,
  isConnected,
  onStartRecording,
  onStopRecording,
  onReset
}: ControlOverlayProps) {
  const [taskName, setTaskName] = useState('');
  const [showRecordDialog, setShowRecordDialog] = useState(false);

  const handleStartRecording = () => {
    setShowRecordDialog(true);
  };

  const handleConfirmRecord = () => {
    if (taskName.trim()) {
      onStartRecording(taskName);
      setShowRecordDialog(false);
      setTaskName('');
    }
  };

  const handleCancelRecord = () => {
    setShowRecordDialog(false);
    setTaskName('');
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="control-overlay">
      {/* Connection Status */}
      <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
        {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
      </div>

      {/* Control Hints */}
      <div className="control-hints">
        <h3>🎮 Manual Control</h3>

        {robotType === 'mobile' && (
          <div className="controls-guide">
            <div className="control-group">
              <div className="control-label">Movement</div>
              <div className="key-row">
                <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
              </div>
              <div className="control-desc">Forward / Left / Back / Right</div>
            </div>

            <div className="control-group">
              <div className="control-label">Rotation</div>
              <div className="key-row">
                <kbd>Q</kbd><kbd>E</kbd>
              </div>
              <div className="control-desc">Rotate Left / Right</div>
            </div>
          </div>
        )}

        {robotType === 'arm' && (
          <div className="controls-guide">
            <div className="control-group">
              <div className="control-label">XY Position</div>
              <div className="key-row">
                <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd>
              </div>
              <div className="control-desc">Move End-Effector</div>
            </div>

            <div className="control-group">
              <div className="control-label">Z Height</div>
              <div className="key-row">
                <kbd>N</kbd><kbd>M</kbd>
              </div>
              <div className="control-desc">Up / Down</div>
            </div>

            <div className="control-group">
              <div className="control-label">Rotation</div>
              <div className="key-row">
                <kbd>J</kbd><kbd>K</kbd>
              </div>
              <div className="control-desc">Counter-CW / Clockwise</div>
            </div>

            <div className="control-group">
              <div className="control-label">Gripper</div>
              <div className="key-row">
                <kbd>Space</kbd>
              </div>
              <div className="control-desc">Close (hold) / Open (release)</div>
            </div>
          </div>
        )}

        {robotType === 'drone' && (
          <div className="controls-guide">
            <div className="control-group">
              <div className="control-label">Pitch / Roll</div>
              <div className="key-row">
                <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
              </div>
              <div className="control-desc">Forward / Left / Back / Right</div>
            </div>

            <div className="control-group">
              <div className="control-label">Altitude</div>
              <div className="key-row">
                <kbd>Space</kbd>
              </div>
              <div className="control-desc">Increase Thrust</div>
            </div>

            <div className="control-group">
              <div className="control-label">Yaw</div>
              <div className="key-row">
                <kbd>Q</kbd><kbd>E</kbd>
              </div>
              <div className="control-desc">Rotate Left / Right</div>
            </div>
          </div>
        )}

        {/* Universal Controls */}
        <div className="control-group universal-controls">
          <div className="control-label">Reset</div>
          <div className="key-row">
            <kbd>U</kbd>
          </div>
          <div className="control-desc">Reset to Initial Pose</div>
        </div>
      </div>

      {/* Recording Controls */}
      <div className="recording-controls">
        <h4>📹 Demonstration Recording</h4>

        {!isRecording ? (
          <button
            className="record-btn start"
            onClick={handleStartRecording}
            disabled={!isConnected}
          >
            🔴 Record Demo
          </button>
        ) : (
          <div className="recording-active">
            <div className="recording-indicator">
              <span className="pulse-dot"></span>
              <span>Recording...</span>
            </div>

            <div className="recording-stats">
              <div className="stat">
                <span className="stat-label">Time:</span>
                <span className="stat-value">{formatTime(recordingTime)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Steps:</span>
                <span className="stat-value">{recordingSteps}</span>
              </div>
            </div>

            <button
              className="record-btn stop"
              onClick={onStopRecording}
            >
              ⏹️ Stop Recording
            </button>
          </div>
        )}
      </div>

      {/* Record Dialog */}
      {showRecordDialog && (
        <div className="record-dialog-overlay">
          <div className="record-dialog">
            <h3>Start Recording Demonstration</h3>
            <p>Enter a name for this task demonstration:</p>

            <input
              type="text"
              value={taskName}
              onChange={(e) => setTaskName(e.target.value)}
              placeholder="e.g., pick_and_place, navigate_to_shelf"
              autoFocus
              onKeyPress={(e) => {
                if (e.key === 'Enter' && taskName.trim()) {
                  handleConfirmRecord();
                }
              }}
            />

            <div className="dialog-actions">
              <button className="btn-cancel" onClick={handleCancelRecord}>
                Cancel
              </button>
              <button
                className="btn-confirm"
                onClick={handleConfirmRecord}
                disabled={!taskName.trim()}
              >
                Start Recording
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Reset Button (optional extra) */}
      {onReset && (
        <button
          className="reset-btn"
          onClick={onReset}
          disabled={!isConnected}
        >
          🔄 Reset Robot
        </button>
      )}
    </div>
  );
}
