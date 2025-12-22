Machines
========

Base classes
------------

.. autoclass:: pyRadPlan.machines.Machine
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyRadPlan.machines.ExternalBeamMachine
   :members:
   :undoc-members:
   :show-inheritance:

Photons
-------

.. autoclass:: pyRadPlan.machines.PhotonLINAC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyRadPlan.machines.PhotonSVDKernel
    :members:
    :undoc-members:
    :show-inheritance:

Ions
----

.. autoclass:: pyRadPlan.machines.IonAccelerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyRadPlan.machines.IonPencilBeamKernel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyRadPlan.machines.LateralCutOff
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyRadPlan.machines._ions.IonBeamFocus
    :members:
    :undoc-members:
    :show-inheritance:


Loading and validating
----------------------

.. autofunction:: pyRadPlan.machines.load_machine
.. autofunction:: pyRadPlan.machines.load_machine_from_mat
.. autofunction:: pyRadPlan.machines.load_from_name
.. autofunction:: pyRadPlan.machines.validate_machine
