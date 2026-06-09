class PID():
    def __init__( self, dt ):
        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        
        self.reference = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.output = 0.0
        self.dt = dt
        
        self.integralLimit = 0.0
        self.outputLimit = 0.0
    
    def setKp( self, kp ):
        self.kp = kp
    
    def setKi( self, ki ):
        self.ki = ki
        
    def setKd( self, kd ):
        self.kd = kd
    
    def setDt( self, dt ):
        self.dt = dt
        
    def setIntegralLimit( self, integralLimit ):
        self.integralLimit = integralLimit
    
    def setOutputLimit( self, outputLimit ):
        self.outputLimit = outputLimit
    
    def update( self, reference, actual ):
        self.reference = reference # 设定值
        self.error = self.reference - actual    # 误差
        self.integral += self.error * self.dt   # 积分
        self.integral = max( min( self.integral, self.integralLimit ), -self.integralLimit ) # 积分限幅
        self.derivative = ( self.error - self.prev_error ) / self.dt # 微分
        self.output = ( self.kp * self.error ) + ( self.ki * self.integral ) + ( self.kd * self.derivative ) # PID输出
        self.prev_error = self.error # 更新上次误差
        self.output = float(max( min( self.output, self.outputLimit ), -self.outputLimit )) # 输出限幅
        return self.output
    
    def update_err( self, err ):
        self.error = err
        self.integral += self.error * self.dt   # 积分
        self.integral = max( min( self.integral, self.integralLimit ), -self.integralLimit ) # 积分限幅
        self.derivative = ( self.error - self.prev_error ) / self.dt # 微分
        self.output = ( self.kp * self.error ) + ( self.ki * self.integral ) + ( self.kd * self.derivative ) # PID输出
        self.prev_error = self.error # 更新上次误差
        self.output = float(max( min( self.output, self.outputLimit ), -self.outputLimit )) # 输出限幅
        return self.output
        

    def reset( self ):
        self.error = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.output = 0.0
        