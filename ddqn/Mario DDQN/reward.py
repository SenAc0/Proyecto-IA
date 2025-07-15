import gym

class customReward(gym.Wrapper):
    def __init__(self, env):
        super(customReward, self).__init__(env)
        self.prev_x_pos = None
        self.prev_score = 0
        self.stuck_time = 0
        self.max_x_pos = 0

    def reset(self, **kwargs):
        self.prev_x_pos = None
        self.prev_score = 0
        self.stuck_time = 0
        self.max_x_pos = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Obtener información del estado actual
        curr_x = info.get('x_pos', 0)
        curr_score = info.get('score', 0)
        flag_get = info.get('flag_get', False)

        # Inicializar valores previos en el primer paso
        if self.prev_x_pos is None:
            self.prev_x_pos = curr_x
            self.prev_score = curr_score
        
        # Recompensa por superar el máximo x alcanzado
        if curr_x > self.max_x_pos:
            reward += 2.0
            self.max_x_pos = curr_x
            
        # Recompensa por incremento en score
        score_increase = curr_score - self.prev_score
        if score_increase > 0:
            reward += score_increase * 0.01
            
        # Recompensa masiva por llegar a la bandera
        if flag_get:
            reward += 500.0
        
        # Penalización creciente después de 10 frames sin avanzar
        if curr_x <= self.prev_x_pos:
            self.stuck_time += 1
            if self.stuck_time >= 10:
                reward -= 0.1 * (self.stuck_time - 9)
        else:
            self.stuck_time = 0  

        # Actualizar valores previos
        self.prev_x_pos = curr_x
        self.prev_score = curr_score

        return obs, reward, done, info
