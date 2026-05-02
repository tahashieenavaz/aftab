from .Aftab import Aftab

agent = Aftab(frames=10)
print("Aftab can initialize successfully")

agent.train(environment="Pong-v5", seed=42)
