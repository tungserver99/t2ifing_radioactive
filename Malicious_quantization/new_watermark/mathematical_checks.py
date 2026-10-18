"""Check local SGD/LoRA algebra only; NOT evidence of diffusion radioactivity."""
import torch


def main():
    torch.manual_seed(13)
    dtype=torch.float64
    w=torch.randn(7,6,dtype=dtype)
    a=torch.randn(3,6,dtype=dtype,requires_grad=True)
    b=torch.zeros(7,3,dtype=dtype,requires_grad=True)
    x=torch.randn(6,11,dtype=dtype)
    yc=torch.randn(7,11,dtype=dtype)
    yw=yc+.2*torch.randn_like(yc)
    scale=.7; eta=.001
    def update(target):
        eff=w+scale*b@a
        loss=.5*((eff@x-target)**2).sum()
        g,ga,gb=torch.autograd.grad(loss,(eff,a,b))
        measured=scale*(b-eta*gb)@(a-eta*ga)
        expected=-eta*scale**2*g@a.T@a
        torch.testing.assert_close(ga,torch.zeros_like(a))
        torch.testing.assert_close(measured,expected)
        return measured.detach()
    dw,dc=update(yw),update(yc)
    q=torch.randn_like(w)
    r=lambda z:.5*((z-q)**2).sum()
    actual=r(w+dw)-r(w+dc)
    linear=((w-q)*(dw-dc)).sum()
    bound=.5*(dw.square().sum()+dc.square().sum())
    assert (actual-linear).abs()<=bound+1e-12
    # First-step changes outside the row space of A are inaccessible.
    _,_,vh=torch.linalg.svd(a.detach(),full_matrices=True)
    g_null=torch.randn(7,1,dtype=dtype)@vh[-1:]
    torch.testing.assert_close(g_null@a.T@a,torch.zeros_like(w),atol=1e-12,rtol=0)
    print('PASS: first-step LoRA algebra; local Taylor bound; inaccessible gradient direction.')
    print('These checks validate equations only, not the proposed watermark or a trained model.')


if __name__=='__main__': main()
