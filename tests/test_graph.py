"""CPU checks for directed graph propagation and output refinement."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('geodist', ROOT / 'topox/core/geodist.py')
geodist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geodist)
# Isolate this pure PyTorch module from optional OpenMMLab/CUDA dependencies.
tree = ast.parse((ROOT / 'topox/models/dense_heads/topox_head.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectedTopologyAggregation')
scope = {'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<graph-module>', 'exec'), scope)
DirectedTopologyAggregation = scope['DirectedTopologyAggregation']


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.module = DirectedTopologyAggregation(4)
        nn.init.normal_(self.module.fusion[-1].weight)
        self.query = torch.randn(3, 4)

    def test_empty_edges_are_identity(self):
        actual = self.module(self.query, torch.zeros(3, 3), torch.ones(3, dtype=torch.bool))
        self.assertTrue(torch.equal(actual, self.query))

    def test_invalid_nodes_are_isolated(self):
        valid = torch.tensor([True, True, False])
        edges = torch.tensor([[0., .7, .9], [.2, 0., .8], [.6, .5, 0.]])
        a = self.module(self.query, edges, valid)
        edges[2, :] = 0.; edges[:, 2] = 0.
        b = self.module(self.query, edges, valid)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[2], self.query[2]))

    def test_topk_reordering_preserves_graph_alignment(self):
        edges = torch.rand(3, 3)
        valid = torch.ones(3, dtype=torch.bool)
        order = torch.tensor([2, 0, 1])
        expected = self.module(self.query, edges, valid)[order]
        actual = self.module(self.query[order], edges[order][:, order], valid[order])
        torch.testing.assert_close(actual, expected)

    def test_geodist_direction_diagonal_and_no_mutation(self):
        centers = np.array([[[0, 0, 0], [1, 0, 0]], [[2, 0, 0], [4, 0, 0]]], dtype=np.float32)
        p = np.full((2, 2), .8, dtype=np.float32)
        before = p.copy()
        out = geodist.fuse_geodist(p, centers)
        self.assertEqual(out[0, 0], p[0, 0])
        self.assertAlmostEqual(float(out[0, 1]), float(.8 + np.exp(-1 / 11.5275)), places=6)
        self.assertAlmostEqual(float(out[1, 0]), float(.8 + np.exp(-16 / 11.5275)), places=6)
        self.assertGreater(out[0, 1], 1.)
        np.testing.assert_array_equal(p, before)

    def test_empty_geodist(self):
        self.assertEqual(geodist.fuse_geodist(np.empty((0, 0)), []).shape, (0, 0))


if __name__ == '__main__':
    unittest.main()
