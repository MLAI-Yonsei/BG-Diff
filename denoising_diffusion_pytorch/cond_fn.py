import torch
import torch.functional as F

def classifier_cond_fn(x, t, classifier, y, classifier_scale=1):
    """
    return the graident of the classifier outputing y wrt x.
    formally expressed as d_log(classifier(x, t)) / dx
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
        return grad
    
    # added
def regressor_cond_fn(x, t, regressor, y, g, regressor_scale=1):
    """
    return the gradient of the MSE of the regressor output and y wrt x.
    formally expressed as d_mse(regressor(x, t), y) / dx
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        torch.nn.utils.clip_grad_norm_(x_in, 1.0)
        # predictions, _ = regressor(x_in, t, g)
        predictions = regressor(x_in, t, g)[0]
        mse = ((predictions - y) ** 2).mean()
        grad = torch.autograd.grad(mse, x_in)[0] * regressor_scale
        return grad