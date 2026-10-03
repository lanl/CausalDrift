"""ACD and UnCLe adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .models import ModelConfig, _prepend, _require_import, _third_party_dir, _unit_scale
from .uncle_compat import install_tcn_fallback


class ACDRunner:
    """Train ACD over one evaluation window and return its graph."""

    def __init__(self, config: ModelConfig):
        self.config = config
        _prepend(_third_party_dir(config) / "AmortizedCausalDiscovery" / "codebase")
        try:
            import torch
            from model import model_loader
            from model import utils as acd_utils
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import("acd", "the benchmark", error) from error
        self.torch = torch
        self.model_loader = model_loader
        self.utils = acd_utils

    def _args(self, num_nodes: int, timesteps: int):
        torch = self.torch
        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return SimpleNamespace(
            seed=self.config.seed,
            GPU_to_use=None,
            epochs=100,
            batch_size=128,
            batch_size_multiGPU=128,
            lr=5e-4,
            lr_decay=200,
            gamma=0.5,
            prediction_steps=min(10, max(1, timesteps - 1)),
            encoder_hidden=256,
            decoder_hidden=256,
            encoder="mlp",
            decoder="mlp",
            prior=1,
            edge_types=2,
            global_temp=False,
            load_temperatures=False,
            alpha=2,
            num_cats=3,
            unobserved=0,
            model_unobserved=0,
            teacher_forcing=0,
            suffix="_causaldrift",
            timesteps=timesteps,
            num_atoms=num_nodes,
            dims=1,
            datadir="",
            save_folder="",
            expername="",
            sym_save_folder="",
            load_folder="",
            test_time_adapt=False,
            lr_logits=0.01,
            num_tta_steps=100,
            temp=0.5,
            hard=False,
            var=5e-7,
            encoder_dropout=0.0,
            decoder_dropout=0.0,
            factor=True,
            validate=False,
            shuffle_unobserved=True,
            skip_first=True,
            use_encoder=True,
            cuda=device.type != "cpu",
            no_cuda=device.type == "cpu",
            device=device,
            num_GPU=1 if device.type != "cpu" else None,
        )

    def fit_predict(self, windows: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"ACD expects [window, time, node], got {values.shape}")
        batch, timesteps, nodes = values.shape
        if timesteps < 2:
            raise ValueError("ACD requires at least two time steps")
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        args = self._args(nodes, timesteps)
        data = np.transpose(values[..., None], (0, 2, 1, 3))
        tensor = torch.tensor(np.nan_to_num(data), dtype=torch.float32)
        rel_rec, rel_send = self.utils.create_rel_rec_send(args, nodes)
        encoder, decoder, optimizer, scheduler, _edge_probs = self.model_loader.load_model(args, None, None, None, None)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor), batch_size=min(128, batch), shuffle=True)
        for _ in range(args.epochs):
            encoder.train()
            decoder.train()
            for (batch_values,) in loader:
                batch_values = batch_values.to(args.device).contiguous()
                optimizer.zero_grad()
                logits = encoder(batch_values, rel_rec, rel_send)
                edges = self.utils.gumbel_softmax(logits, tau=args.temp, hard=False)
                probabilities = self.utils.my_softmax(logits, -1)
                output = decoder(batch_values, edges, rel_rec, rel_send, args.prediction_steps)
                loss = self.utils.nll_gaussian(output, batch_values[:, :, 1:, :], args.var)
                loss = loss + self.utils.kl_categorical_uniform(probabilities, args.num_atoms, args.edge_types)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
                    optimizer.step()
            scheduler.step()
        encoder.eval()
        with torch.no_grad():
            logits = encoder(tensor.to(args.device), rel_rec, rel_send)
            scores = self.utils.my_softmax(logits, -1)[..., 1].detach().cpu().numpy()
        output = np.zeros((batch, nodes, nodes), dtype=np.float32)
        receivers, senders = np.where(np.ones((nodes, nodes)) - np.eye(nodes))
        for edge_index, (target, source) in enumerate(zip(receivers, senders)):
            output[:, source, target] = scores[:, edge_index]
        return np.stack([_unit_scale(item) for item in output], axis=0)


class UnCLeRunner:
    """Train upstream UnCLe's VARP model on one window, without logs."""

    def __init__(self, config: ModelConfig):
        self.config = config
        install_tcn_fallback()
        _prepend(_third_party_dir(config) / "uncle-causal-discovery" / "bin")
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from experimental_utils import VARP, compute_l1_loss
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import("uncle", "the benchmark", error) from error
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.VARP = VARP
        self.compute_l1_loss = compute_l1_loss

    def fit_predict(self, windows: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"UnCLe expects [window, time, node], got {values.shape}")
        if values.shape[1] < 2:
            raise ValueError("UnCLe requires at least two time steps")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        batch, _timesteps, nodes = values.shape
        channels = np.transpose(values, (0, 2, 1))
        inputs = torch.tensor(channels[:, :, :-1], dtype=torch.float32)
        targets = torch.tensor(channels[:, :, 1:], dtype=torch.float32)
        model = self.VARP(
            nodes,
            lag=1,
            seed=self.config.seed,
            encoder_layers=6 * [20],
            decoder_layers=6 * [20],
            kernel_size=8,
        ).to(device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=min(16, batch), shuffle=True)
        optimizer = self.optim.Adam(model.parameters(), lr=0.005)
        criterion = self.nn.MSELoss()
        for epoch in range(150):
            reconstruction = epoch < 50
            model.train()
            for input_batch, target_batch in loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                optimizer.zero_grad()
                reconstructed = model(input_batch, False)
                loss = criterion(reconstructed, input_batch)
                regularized = model.get_regularized_params(stage=0 if reconstruction else 1)
                if not reconstruction:
                    loss = loss + criterion(model(input_batch, True), target_batch)
                for parameter in regularized:
                    loss = loss + self.compute_l1_loss(parameter)
                loss.backward()
                optimizer.step()
        model.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed + 1_000_003)
        graphs: list[np.ndarray] = []
        with torch.inference_mode():
            for input_values, target_values in zip(inputs, targets):
                input_values, target_values = input_values.to(device), target_values.to(device)
                shuffled = input_values.repeat(nodes, 1, 1)
                for node in range(nodes):
                    permutation = torch.randperm(shuffled.shape[2], generator=generator, device=device)
                    shuffled[node, node] = shuffled[node, node, permutation]
                full_prediction = model(input_values.unsqueeze(0), True)[0]
                reduced_prediction = torch.cat([model(shuffled[start : start + 8], True) for start in range(0, nodes, 8)])
                full_error = (full_prediction - target_values).norm(dim=1, p=2)
                reduced_error = (reduced_prediction - target_values.repeat(nodes, 1, 1)).norm(dim=2, p=2)
                graph = (reduced_error - full_error).clamp_min(0.0)
                graphs.append(_unit_scale(graph.detach().cpu().numpy()))
        return np.stack(graphs, axis=0)
