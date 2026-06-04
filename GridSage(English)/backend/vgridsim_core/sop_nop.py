# file: sop_nop.py
"""
This module defines the data classes of soft switch (SOP) and normally open point (NOP).

These two classes are mainly used as data containers to store data read from configuration files (such as asexcel files).
Physical parameters of SOP and NOP devices. The simulation and optimization modules will use instances of these classes to build grid models.
"""

from typing import Optional
from xml.etree.ElementTree import Element
from fpowerkit.utils import NFloat


class SOP:
    """
    Soft Open Point (SOP) class.

    SOP is an advanced power electronic device used for flexible power flow control in distribution networks.
    It is usually seen as an ideal controllable source that can connect two different feeders or buses,
    And accurately inject or absorb specified amounts of active and reactive power to achieve load balancing, voltage regulation and other functions.
    """

    def __init__(self, id: str, bus1: str, bus2: str, p_max_pu: float, q_max_pu: float, loss_coeff: float = 0.05,
                 active: bool = True):
        """
        Initialize the SOP object.

        Parameters:
            id (str): unique identifier of SOP.
            bus1 (str): ID of the first connected bus.
            bus2 (str): ID of the second bus connected.
            p_max_pu (float): Maximum active power (per unit) that can be transmitted or absorbed.
            q_max_pu (float): Maximum reactive power (per unit) that can be transmitted or absorbed.
            loss_coeff (float): Power loss coefficient, used to estimate the operating loss of the SOP itself.
            active (bool): Mark whether this SOP is available in the current simulation.
        """
        self._id = id
        self._bus1 = bus1
        self._bus2 = bus2
        self._p_max = p_max_pu
        self._q_max = q_max_pu
        self._loss_coeff = loss_coeff
        self.active = active

        # The following properties are used to store results after the solver is run
        self.P1 = None # Active power on bus1 side (pu)
        self.Q1 = None # Reactive power on bus1 side (pu)
        self.P2 = None # Active power on bus2 side (pu)
        self.Q2 = None # Reactive power on bus2 side (pu)

    @property
    def ID(self) -> str:
        """Get the unique identifier ID of the SOP."""
        return self._id

    @property
    def Bus1(self) -> str:
        """Get the ID of the first bus connected to the SOP."""
        return self._bus1

    @property
    def Bus2(self) -> str:
        """Get the ID of the second bus connected to the SOP."""
        return self._bus2

    @property
    def PMax(self) -> float:
        """Get the maximum active power transmission capability (pu) of the SOP."""
        return self._p_max

    @property
    def QMax(self) -> float:
        """Get the maximum reactive power transmission capability (pu) of the SOP."""
        return self._q_max

    @property
    def LossCoeff(self) -> float:
        """Get the loss coefficient of SOP."""
        return self._loss_coeff

    def __repr__(self) -> str:
        """Returns the string representation of the object, which is convenient for printing and viewing during debugging."""
        return f"SOP(id='{self.ID}', bus1='{self.Bus1}', bus2='{self.Bus2}', p_max_pu={self.PMax}, q_max_pu={self.QMax}, active={self.active})"


class NOP:
    """
    Normally Open Point (NOP) class.

    NOP ​​is a tie switch in the distribution network, which is open during normal operation.
    When network reconstruction is required to achieve failure recovery, load balancing or Grid Loss optimization,
    You can change the topology of the network by closing one or more NOPs.
    """

    def __init__(self, id: str, bus1: str, bus2: str, r_pu: float, x_pu: float, max_I_kA: float = float('inf'),
                 active: bool = False):
        """
        Initialize the NOP object.

        Parameters:
            id (str): The unique identifier of the NOP.
            bus1 (str): ID of the first connected bus.
            bus2 (str): ID of the second bus connected.
            r_pu (float): When the NOP is closed, the equivalent line resistance (unit value).
            x_pu (float): When the NOP is closed, the reactance of the equivalent line (unit value).
            max_I_kA (float): The maximum current allowed to pass when the NOP is closed (kA).
            active (bool): The initial state of NOP, False means open (default), True means closed.
        """
        self._id = id
        self._bus1 = bus1
        self._bus2 = bus2
        self._r = r_pu
        self._x = x_pu
        self._max_I = max_I_kA
        self.active = active # This state will be determined by the decision variable during the optimization process

        # The following properties are used to store results after the solver is run
        self.P = None # Active power flowing when closed (pu)
        self.Q = None # Reactive power flowing when closed (pu)
        self.I = None # Current flowing when closed (pu)

    @property
    def ID(self) -> str:
        """Get the unique identifier ID of the NOP."""
        return self._id

    @property
    def Bus1(self) -> str:
        """Get the ID of the first bus connected to the NOP."""
        return self._bus1

    @property
    def Bus2(self) -> str:
        """Get the ID of the second bus connected to the NOP."""
        return self._bus2

    @property
    def R(self) -> float:
        """Get the equivalent resistance (pu) when the NOP is closed."""
        return self._r

    @property
    def X(self) -> float:
        """Get the equivalent reactance (pu) when the NOP is closed."""
        return self._x

    @property
    def MaxI(self) -> float:
        """Get the maximum current capacity of the NOP (kA)."""
        return self._max_I

    def __repr__(self) -> str:
        """Returns the string representation of the object, which is convenient for printing and viewing during debugging."""
        return f"NOP(id='{self.ID}', bus1='{self.Bus1}', bus2='{self.Bus2}', r_pu={self.R}, x_pu={self.X}, max_I_kA={self.MaxI}, active={self.active})"