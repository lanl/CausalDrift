# Optional upstream model sources

Third-party model implementations are not included. Clone the five upstream
projects below into this directory when running `acd`, `cuts`, `cdans`,
`kausal`, or `uncle`. The hashes identify the revisions used for this benchmark.

```bash
git clone https://github.com/loeweX/AmortizedCausalDiscovery.git AmortizedCausalDiscovery
git -C AmortizedCausalDiscovery checkout 4d1661118c9fcfbefbbd00226bc41f7a76743e11

git clone https://github.com/jarrycyx/UNN.git UNN
git -C UNN checkout 2212724cce8a7752db564089a7537fecedb0e36d

git clone https://github.com/hferdous/CDANs.git CDANs
git -C CDANs checkout dcf7533c5115a86a4cc8ec02425a5786c445a854

git clone https://github.com/juannat7/kausal.git kausal
git -C kausal checkout 4db67785a7230f8e8b6c6e3fb05bfd1298f42074

git clone https://github.com/etigerstudio/uncle-causal-discovery.git uncle-causal-discovery
git -C uncle-causal-discovery checkout 820da2a7690e8528ecf8d346cd44545624dfe85b
```

Install the upstream requirements as needed, then point the CLI at this
directory with `--third-party-dir third_party`. The internal UnCLe adapter has
a small `TemporalConvNet` compatibility fallback, so `tsai`/`fastai` is not
required solely for this benchmark.

Before public redistribution, review each upstream project's license and
citation requirements. This repository only imports them at runtime.
