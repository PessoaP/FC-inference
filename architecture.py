import torch
import normflows as nf

latent_size = 1
context_size = 4

def make_model(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               context_size=4,
               hidden_layers=5,
               spline_hidden_dims = [64, 128, 256, 512],
               tail_bound = 30,
               conditional=True):
        


    if conditional:
        q0 = nf.distributions.DiagGaussian(1)

        flows = []
        for hidden_dim in spline_hidden_dims:
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_dim, num_context_channels=context_size, tail_bound=tail_bound
            ))
            flows.append(nf.flows.LULinearPermute(latent_size))

        model = nf.ConditionalNormalizingFlow(q0, flows)

    else:
        q0 = nf.distributions.DiagGaussian(1,trainable=False)
        flows = []
        for hidden_dim in spline_hidden_dims:
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_dim, tail_bound=tail_bound
            ))
            flows.append(nf.flows.LULinearPermute(latent_size))

        model = nf.ConditionalNormalizingFlow(q0, flows)
        model=nf.NormalizingFlow(q0,flows)


    return model.to(device)

