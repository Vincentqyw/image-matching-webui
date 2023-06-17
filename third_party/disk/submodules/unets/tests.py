import torch, unittest
from unets import thin_setup, fat_setup, Unet, ThinUnetUpBlock, \
                  ThinUnetDownBlock, AttentionGate

class BaseTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64], up=[40, 4]
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 24, 24]), output.size())

    def test_inequal_output_symmetric(self):
        unet = Unet(
            down=[16, 32, 64], up=[40, 1]
        )
        input = torch.zeros(2, 1, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 1, 24, 24]), output.size())

class CheckpointedTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64], up=[40, 4],
            setup={**fat_setup, 'checkpointed': True}
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 24, 24]), output.size())

class NoBiasTests(unittest.TestCase):
    def test_bias(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64], up=[40, 4],
        )
        checker = lambda name_weight: 'bias' in name_weight[0]
        bias = any(map(checker, unet.named_parameters()))
        self.assertTrue(bias)

    def test_no_bias(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64], up=[40, 4],
            setup={**fat_setup, 'bias': False}
        )
        checker = lambda name_weight: 'bias' not in name_weight[0]
        no_bias = all(map(checker, unet.named_parameters()))
        self.assertTrue(no_bias)

class ThinTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64],
            up=[40, 4],
            setup=thin_setup
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 64, 64]), output.size())

    def test_inequal_output_symmetric(self):
        unet = Unet(
            down=[16, 32, 64],
            up=[40, 1],
            setup=thin_setup
        )
        input = torch.zeros(2, 1, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 1, 64, 64]), output.size())

class AttentionTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64],
            up=[40, 4],
            setup={**thin_setup, 'gate': AttentionGate}
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 64, 64]), output.size())

    def test_inequal_output_symmetric(self):
        unet = Unet(
            down=[16, 32, 64],
            up=[40, 1],
            setup={**thin_setup, 'gate': AttentionGate}
        )
        input = torch.zeros(2, 1, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 1, 64, 64]), output.size())

unittest.main()
