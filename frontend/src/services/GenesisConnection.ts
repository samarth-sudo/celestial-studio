/**
 * Genesis Connection Manager
 * Manages WebSocket connection, state synchronization, and API calls to Genesis backend
 */

export interface GenesisConfig {
  backend?: string;
  fps?: number;
  render_width?: number;
  render_height?: number;
  stream_quality?: 'draft' | 'medium' | 'high' | 'ultra';
}

export interface RobotConfig {
  robot_id: string;
  robot_type: 'mobile' | 'arm' | 'drone' | 'franka' | 'go2';
  position?: [number, number, number];
}

export interface ObstacleConfig {
  obstacle_id: string;
  position: [number, number, number];
  size: [number, number, number];
}

export interface SimulationState {
  timestamp: number;
  step: number;
  fps: number;
  robots: Record<string, RobotState>;
  obstacles: Record<string, ObstacleState>;
}

export interface RobotState {
  position: [number, number, number];
  orientation: [number, number, number, number];
  velocity: [number, number, number];
}

export interface ObstacleState {
  position: [number, number, number];
  size: [number, number, number];
}

export interface StreamStats {
  runtime_seconds: number;
  frames_delivered: number;
  bytes_delivered: number;
  avg_fps: number;
  avg_bitrate_mbps: number;
}

export type MessageHandler = (message: any) => void;
export type ConnectionHandler = (connected: boolean) => void;

export class GenesisConnection {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private connected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectDelay: number = 2000;

  private messageHandlers: Map<string, MessageHandler[]> = new Map();
  private connectionHandlers: ConnectionHandler[] = [];

  private heartbeatInterval: NodeJS.Timeout | null = null;
  private heartbeatTimeout: number = 30000; // 30 seconds

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  /**
   * Initialize Genesis simulation
   */
  async initialize(config: GenesisConfig = {}): Promise<any> {
    try {
      // Check status first
      const status = await this.getStatus();
      if (!status.available) {
        throw new Error('Genesis not available on backend');
      }

      // Initialize with config
      const response = await fetch(`${this.baseUrl}/api/genesis/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backend: config.backend || 'auto',
          fps: config.fps || 60,
          render_width: config.render_width || 1920,
          render_height: config.render_height || 1080,
          stream_quality: config.stream_quality || 'medium',
        }),
      });

      if (!response.ok) {
        throw new Error(`Initialization failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Genesis initialized:', result);
      return result;
    } catch (error) {
      console.error('❌ Initialization error:', error);
      throw error;
    }
  }

  /**
   * Get Genesis status
   */
  async getStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/genesis/status`);
    return response.json();
  }

  /**
   * Add robot to simulation
   */
  async addRobot(config: RobotConfig): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/genesis/robot/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        robot_id: config.robot_id,
        robot_type: config.robot_type,
        position: config.position || [0, 0, 0.5],
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to add robot: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Add obstacle to simulation
   */
  async addObstacle(config: ObstacleConfig): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/genesis/obstacle/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`Failed to add obstacle: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Build scene (required before starting simulation)
   */
  async buildScene(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/genesis/scene/build`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to build scene: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Control simulation
   */
  async control(action: 'start' | 'stop' | 'reset' | 'step'): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/genesis/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });

    if (!response.ok) {
      throw new Error(`Control action failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get latest frame as JPEG
   */
  async getFrame(): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/genesis/stream/frame`);

    if (!response.ok) {
      throw new Error(`Failed to get frame: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * Get stream statistics
   */
  async getStreamStats(): Promise<StreamStats | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/genesis/stream/stats`);
      const data = await response.json();

      if (data.status === 'running') {
        return data.stats;
      }

      return null;
    } catch (error) {
      console.error('Failed to get stream stats:', error);
      return null;
    }
  }

  /**
   * Connect to WebSocket
   */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    const wsUrl = this.baseUrl
      .replace('http://', 'ws://')
      .replace('https://', 'wss://');

    console.log('🔌 Connecting to WebSocket:', `${wsUrl}/api/genesis/ws`);

    this.ws = new WebSocket(`${wsUrl}/api/genesis/ws`);

    this.ws.onopen = () => {
      console.log('🔌 WebSocket connected');
      this.connected = true;
      this.reconnectAttempts = 0;
      this.notifyConnectionChange(true);
      this.startHeartbeat();
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('🔌 WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      this.connected = false;
      this.notifyConnectionChange(false);
      this.stopHeartbeat();
      this.attemptReconnect();
    };
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.connected = false;
    this.notifyConnectionChange(false);
  }

  /**
   * Send message to WebSocket
   */
  send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  /**
   * Send control command via WebSocket
   */
  sendControl(action: 'start' | 'stop' | 'reset'): void {
    this.send({
      type: 'control',
      action: action,
    });
  }

  /**
   * Request frame via WebSocket
   */
  requestFrame(): void {
    this.send({
      type: 'get_frame',
    });
  }

  /**
   * Register message handler
   */
  on(messageType: string, handler: MessageHandler): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }

    this.messageHandlers.get(messageType)!.push(handler);
  }

  /**
   * Unregister message handler
   */
  off(messageType: string, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(messageType);

    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * Register connection change handler
   */
  onConnectionChange(handler: ConnectionHandler): void {
    this.connectionHandlers.push(handler);
  }

  /**
   * Unregister connection change handler
   */
  offConnectionChange(handler: ConnectionHandler): void {
    const index = this.connectionHandlers.indexOf(handler);
    if (index > -1) {
      this.connectionHandlers.splice(index, 1);
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(message: any): void {
    const messageType = message.type;

    if (!messageType) {
      console.warn('Message without type:', message);
      return;
    }

    // Notify handlers
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }

    // Also notify wildcard handlers
    const wildcardHandlers = this.messageHandlers.get('*');
    if (wildcardHandlers) {
      wildcardHandlers.forEach((handler) => handler(message));
    }
  }

  /**
   * Notify connection change
   */
  private notifyConnectionChange(connected: boolean): void {
    this.connectionHandlers.forEach((handler) => handler(connected));
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(
      `🔄 Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
    );

    setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, this.heartbeatTimeout);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

// Singleton instance
let genesisConnection: GenesisConnection | null = null;

/**
 * Get or create Genesis connection singleton
 */
export function getGenesisConnection(baseUrl?: string): GenesisConnection {
  if (!genesisConnection) {
    genesisConnection = new GenesisConnection(baseUrl);
  }
  return genesisConnection;
}

/**
 * Reset Genesis connection singleton
 */
export function resetGenesisConnection(): void {
  if (genesisConnection) {
    genesisConnection.disconnect();
    genesisConnection = null;
  }
}

export default GenesisConnection;
