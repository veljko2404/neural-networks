from abc import abstractmethod

from backend.backend import np
from layers.function import Function


class LossFunction(Function):

    def __init__(self, name: str = 'loss'):
        super().__init__(name)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented!")

    @abstractmethod
    def __call__(self, y: np.ndarray, t: np.ndarray) -> float:
        """Call će vraćati vrednost funkcije greške"""
        pass

    @abstractmethod
    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Od ovog 'sloja' će počinjati prolazak unazad. Python dozvoljava da metodu preklopimo
           sa argumentom drugačijeg naziva, pa je za razliku od ostalih slojeva ulazni ndimenzionalni niz
           ovde nazvan target (a ne dEdO, kao kod ostalih slojeva), da asocira na to da će se kao argument
           prosleđivati 'ciljne' vrednosti koje želimo da naša mreža nauči da na izlazu generiše.

           Rezultat poziva su parcijalni izvodi dEdy
        """
        pass
