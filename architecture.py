import torch
import normflows as nf

latent_size = 1
context_size = 5


def make_model(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),hidden_layers=5):
        
    q0 = nf.distributions.DiagGaussian(1)

    flows = [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, 64,num_context_channels=context_size,tail_bound=30),
             nf.flows.LULinearPermute(latent_size),
             nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, 128,num_context_channels=context_size,tail_bound=30),
             nf.flows.LULinearPermute(latent_size),
             nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, 256,num_context_channels=context_size,tail_bound=30),
             nf.flows.LULinearPermute(latent_size),      
             nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, 512,num_context_channels=context_size,tail_bound=30),
             nf.flows.LULinearPermute(latent_size)
             ]

    target = torch.distributions.Normal(0,1)
    model = nf.ConditionalNormalizingFlow(q0, flows, target)


    return model.to(device)