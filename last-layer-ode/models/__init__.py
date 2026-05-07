from models.ode_rnn import OdeRNN
from models.ode_rnn_analytical import AnalyticalOdeRNN
from models.ode_rnn_txtl import TXTLAnalyticalOdeRNN, TXTLAnalyticalBleachOdeRNN, TXTLApproxOdeRNN, TXTLMaturationApproxOdeRNN
from models.bob_gru_verbatim import BobGRUVerbatim
from models.neural_ode import NeuralODE
from models.ode_transformer import OdeTransformer
from models.ode_fixed_theta import OdeFixedTheta
from models.ode_fixed_theta_nn import NeuralOdeCorrection
from models.ode_sample_theta import OdeSampleTheta
from models.neural_ode_gru import NeuralOdeGRU
from models.ode_mamba import OdeMamba
from models.ode_lstm import OdeLSTM


try:
    from models.ode_mamba_ssm import OdeMambaSSM
    _mamba_ssm_available = True
except ImportError:
    _mamba_ssm_available = False

try:
    from models.ode_mambapy import OdeMambapySSM
    _mambapy_available = True
except ImportError:
    _mambapy_available = False

MODELS: dict = {
    "ode_rnn":            OdeRNN,
    "lstm_rnn":           OdeLSTM,
    "ode_rnn_analytical": AnalyticalOdeRNN,
    "ode_rnn_txtl":       TXTLAnalyticalOdeRNN,
    "ode_rnn_txtl_approx":     TXTLApproxOdeRNN,
    "ode_rnn_txtl_mat_approx": TXTLMaturationApproxOdeRNN,
    "ode_rnn_txtl_bleach": TXTLAnalyticalBleachOdeRNN,
    "neural_ode":         NeuralODE,
    "neural_ode_mlp":     NeuralODE,       # alias — same class, explicit name for ablation table
    "ode_transformer":    OdeTransformer,
    "ode_fixed_theta":    OdeFixedTheta,
    "neural_ode_correction": NeuralOdeCorrection,
    "ode_sample_theta":   OdeSampleTheta,
    "neural_ode_gru":     NeuralOdeGRU,
    "bob_gru_verbatim": BobGRUVerbatim,
    # "ode_mamba":          OdeMamba,
    # "ode_transformer_kvcache": OdeTransformer_transformer,
    **({"ode_mamba_ssm": OdeMambaSSM} if _mamba_ssm_available else {}),
    **({"ode_mambapy": OdeMambapySSM} if _mambapy_available else {}),
}
