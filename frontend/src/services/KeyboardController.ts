/**
 * Keyboard Controller for Genesis Teleoperation
 *
 * Captures keyboard input and sends to backend via WebSocket
 * Runs at 60Hz for smooth robot control
 *
 * Keyboard Layout:
 * - WASD: Forward/Left/Back/Right
 * - QE: Rotate Left/Right
 * - Arrow Keys: XY movement (for arms)
 * - NM: Z movement Up/Down
 * - JK: Rotation
 * - Space: Gripper/Altitude
 * - U: Reset robot
 */

export interface KeyboardState {
  // Movement keys
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;

  // Rotation keys
  q: boolean;
  e: boolean;

  // Arrow keys
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;

  // Z movement
  n: boolean;
  m: boolean;

  // Rotation (arms)
  j: boolean;
  k: boolean;

  // Action keys
  space: boolean;
  u: boolean;  // Reset
}

export interface TeleopUpdate {
  type: 'teleop_update';
  robot_id: string;
  result: {
    action: number[];
    state: any;
    recording: boolean;
    steps_recorded?: number;
  };
}

export interface RecordingStatus {
  recording: boolean;
  task_name?: string;
  num_steps: number;
  elapsed_time: number;
}

export class KeyboardController {
  private pressedKeys: Map<string, boolean> = new Map();
  private ws: WebSocket | null = null;
  private sendInterval: number | null = null;
  private robotId: string;
  private robotType: 'mobile' | 'arm' | 'drone';
  private isConnected: boolean = false;

  // Callbacks
  private onUpdateCallback?: (update: TeleopUpdate) => void;
  private onRecordingStartCallback?: (taskName: string) => void;
  private onRecordingStopCallback?: (result: any) => void;
  private onConnectCallback?: () => void;
  private onDisconnectCallback?: () => void;

  constructor(
    wsUrl: string,
    robotId: string = 'robot-1',
    robotType: 'mobile' | 'arm' | 'drone' = 'mobile'
  ) {
    this.robotId = robotId;
    this.robotType = robotType;

    console.log(`🎮 Keyboard Controller initializing for ${robotType} robot`);

    this.connectWebSocket(wsUrl);
    this.setupKeyboardListeners();
  }

  private connectWebSocket(wsUrl: string) {
    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('✅ Teleoperation WebSocket connected');
        this.isConnected = true;
        this.startSendingKeyState();

        if (this.onConnectCallback) {
          this.onConnectCallback();
        }
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleServerMessage(data);
      };

      this.ws.onerror = (error) => {
        console.error('❌ Teleoperation WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('🔌 Teleoperation WebSocket disconnected');
        this.isConnected = false;
        this.stopSendingKeyState();

        if (this.onDisconnectCallback) {
          this.onDisconnectCallback();
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }

  private handleServerMessage(data: any) {
    switch (data.type) {
      case 'teleop_update':
        if (this.onUpdateCallback) {
          this.onUpdateCallback(data as TeleopUpdate);
        }
        break;

      case 'recording_started':
        console.log('🔴 Recording started:', data.result);
        if (this.onRecordingStartCallback) {
          this.onRecordingStartCallback(data.result.task_name);
        }
        break;

      case 'recording_stopped':
        console.log('✅ Recording stopped:', data.result);
        if (this.onRecordingStopCallback) {
          this.onRecordingStopCallback(data.result);
        }
        break;

      case 'robot_reset':
        console.log('🔄 Robot reset');
        break;

      case 'pong':
        // Heartbeat response
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  }

  private setupKeyboardListeners() {
    // Keydown handler
    window.addEventListener('keydown', (e) => {
      const key = e.key.toLowerCase();

      // Don't capture keys if user is typing in an input
      if (e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Prevent default for control keys to avoid page scrolling
      if (['w', 'a', 's', 'd', 'q', 'e', ' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
        e.preventDefault();
      }

      this.pressedKeys.set(key, true);
    });

    // Keyup handler
    window.addEventListener('keyup', (e) => {
      const key = e.key.toLowerCase();
      this.pressedKeys.set(key, false);
    });

    // Clear all keys when window loses focus
    window.addEventListener('blur', () => {
      this.pressedKeys.clear();
    });
  }

  private startSendingKeyState() {
    // Send key state at 60Hz for smooth control
    this.sendInterval = window.setInterval(() => {
      if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.sendKeyState();
      }
    }, 16); // ~60 FPS
  }

  private stopSendingKeyState() {
    if (this.sendInterval !== null) {
      clearInterval(this.sendInterval);
      this.sendInterval = null;
    }
  }

  private sendKeyState() {
    const keyState: KeyboardState = {
      w: this.pressedKeys.get('w') || false,
      a: this.pressedKeys.get('a') || false,
      s: this.pressedKeys.get('s') || false,
      d: this.pressedKeys.get('d') || false,
      q: this.pressedKeys.get('q') || false,
      e: this.pressedKeys.get('e') || false,
      up: this.pressedKeys.get('arrowup') || false,
      down: this.pressedKeys.get('arrowdown') || false,
      left: this.pressedKeys.get('arrowleft') || false,
      right: this.pressedKeys.get('arrowright') || false,
      n: this.pressedKeys.get('n') || false,
      m: this.pressedKeys.get('m') || false,
      j: this.pressedKeys.get('j') || false,
      k: this.pressedKeys.get('k') || false,
      space: this.pressedKeys.get(' ') || false,
      u: this.pressedKeys.get('u') || false,
    };

    // Send keyboard input to backend
    this.send({
      type: 'keyboard_input',
      robot_id: this.robotId,
      robot_type: this.robotType,
      keys: keyState
    });

    // Handle reset key (U)
    if (keyState.u) {
      this.resetRobot();
      // Clear the key to avoid repeated resets
      this.pressedKeys.set('u', false);
    }
  }

  private send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  // Public API

  startRecording(taskName: string) {
    console.log(`🔴 Starting recording: ${taskName}`);
    this.send({
      type: 'start_recording',
      task_name: taskName
    });
  }

  stopRecording() {
    console.log('⏹️ Stopping recording');
    this.send({
      type: 'stop_recording'
    });
  }

  resetRobot() {
    console.log('🔄 Resetting robot');
    this.send({
      type: 'reset_robot',
      robot_id: this.robotId
    });
  }

  getRecordingStatus() {
    this.send({
      type: 'get_recording_status'
    });
  }

  // Callback registration

  onUpdate(callback: (update: TeleopUpdate) => void) {
    this.onUpdateCallback = callback;
  }

  onRecordingStart(callback: (taskName: string) => void) {
    this.onRecordingStartCallback = callback;
  }

  onRecordingStop(callback: (result: any) => void) {
    this.onRecordingStopCallback = callback;
  }

  onConnect(callback: () => void) {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback;
  }

  // Cleanup

  destroy() {
    console.log('🎮 Keyboard Controller cleanup');

    this.stopSendingKeyState();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    // Note: We don't remove window event listeners since they're global
    // If you need to remove them, store references and call removeEventListener
  }

  // Status getters

  get connected(): boolean {
    return this.isConnected;
  }

  getCurrentKeyState(): KeyboardState {
    return {
      w: this.pressedKeys.get('w') || false,
      a: this.pressedKeys.get('a') || false,
      s: this.pressedKeys.get('s') || false,
      d: this.pressedKeys.get('d') || false,
      q: this.pressedKeys.get('q') || false,
      e: this.pressedKeys.get('e') || false,
      up: this.pressedKeys.get('arrowup') || false,
      down: this.pressedKeys.get('arrowdown') || false,
      left: this.pressedKeys.get('arrowleft') || false,
      right: this.pressedKeys.get('arrowright') || false,
      n: this.pressedKeys.get('n') || false,
      m: this.pressedKeys.get('m') || false,
      j: this.pressedKeys.get('j') || false,
      k: this.pressedKeys.get('k') || false,
      space: this.pressedKeys.get(' ') || false,
      u: this.pressedKeys.get('u') || false,
    };
  }
}
