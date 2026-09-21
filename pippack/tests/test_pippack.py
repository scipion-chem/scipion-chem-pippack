import os

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportPdb
from pwchem.protocols import ProtChemPrepareReceptor

from .. import Plugin
from ..protocols import ProtPIPPack
from ..utils import assertHandle

class TestPPIPack(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

        cls._runImportPDB()
        cls._runTargetPreparation()

    @classmethod
    def _runImportPDB(cls):
        protImportPDB = cls.newProtocol(
            ProtImportPdb,
            inputPdbData=0, pdbId='9t5q')
        cls.proj.launchProtocol(protImportPDB, wait=True)
        cls.protImportPDB = protImportPDB

    @classmethod
    def _runTargetPreparation(cls):
        protPrepTarget = cls.newProtocol(
            ProtChemPrepareReceptor,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            rchains=True,
            chain_name='{"model": 0, "chain": "I", "residues": 126}')
        cls.proj.launchProtocol(protPrepTarget, wait=True)
        cls.protPrepTarget = protPrepTarget


    def _runPPIPack(self):
        protPPIPack = self.newProtocol(ProtPIPPack,
                   inputStruct=self.protPrepTarget.outputStructure)

        self.proj.launchProtocol(protPPIPack, wait=True)
        return protPPIPack

    def test(self):
        protPPIPack = self._runPPIPack()
        self._waitOutput(protPPIPack, 'outputAtomStruct', sleepTime=5)

        assertHandle(self.assertIsNotNone, getattr(protPPIPack, 'outputAtomStruct', None), cwd=protPPIPack.getWorkingDir())