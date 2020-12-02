# coding=utf8
import torch


# import data.dict_spk2idx[as dict

class Beam(object):
    def __init__(self, size, dict_spk2idx, n_best=1, cuda=True):
        self.dict_spk2idx = dict_spk2idx
        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(dict_spk2idx['<EOS>'])]
        self.nextYs[0][0] = dict_spk2idx['<BOS>']
        # Has EOS topped the beam yet.
        self._eos = dict_spk2idx['<EOS>']
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # The last hiddens(matrix) for each time.
        self.hiddens = []

        # The last hiddens(matrix) for each time.
        self.sch_hiddens = []

        # The last embs(matrix) for each time.
        self.embs = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

    def updates_sch_embeddings(self, hiddens_this_step):
        if len(self.sch_hiddens) == 0:
            self.sch_hiddens.append([hiddens_this_step])
        else:
            self.sch_hiddens.append(self.sch_hiddens[-1] + [hiddens_this_step])

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut, hidden, emb):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)  
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))
        self.hiddens.append(hidden.index_select(0, prevK))
        self.updates_sch_embeddings(hidden.index_select(0, prevK))
        self.embs.append(emb.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is '<EOS>' and no global score.
        if self.nextYs[-1][0] == self.dict_spk2idx['<EOS>']:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def beam_update(self, state, idx):
        positions = self.getCurrentOrigin()
        for e in state:
            a, br, d = e.size()
            e = e.view(a, self.size, br // self.size, d)
            sentStates = e[:, :, idx]
            sentStates.data.copy_(sentStates.data.index_select(1, positions))

    def beam_update_context(self, state, idx):
        positions = self.getCurrentOrigin()
        e = state.unsqueeze(0)
        a, br, len, d, = e.size()
        e = e.view(a, self.size, br // self.size, len, d)
        sentStates = e[:, :, idx]
        sentStates.data.copy_(sentStates.data.index_select(1, positions))

    def beam_update_hidden(self, state, idx):
        positions = self.getCurrentOrigin()
        e = state
        a, br, d = e.size()
        e = e.view(a, self.size, br // self.size, d)
        sentStates = e[:, :, idx]
        sentStates.data.copy_(sentStates.data.index_select(1, positions))

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn, hidden, emb = [], [], [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            hidden.append(self.hiddens[j][k])
            emb.append(self.embs[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1]), torch.stack(hidden[::-1]), torch.stack(emb[::-1])
