# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:12:02 2024

ComboInjector class for managing combos, actions, and special moves
in a fighting game environment. Supports configurable probabilities for
different action categories (jump/movement/basic/combos).

Author: ruhe
"""

from collections import deque
import numpy as np

class ComboInjector(object):
    """
    A utility class for injecting combos, special moves, and basic actions
    for a fighting game environment. Currently supports 'sfiii' with multi-discrete
    action spaces.

    Attributes
    ----------
    environment_name : str
        Name of the environment (default 'sfiii').
    mode : str
        Action mode (must be 'multi_discrete' currently).
    frame_skip : int
        Frame skip value that can affect certain hold/charge durations.
    characters : dict
        Nested dictionary mapping environment -> {character -> {move definitions}}.
        Each move definition has a 'prob' for sampling, plus 'combo_str' specifying
        how to generate that move sequence.
    base_movement_names : dict
        Mapping of short string direction keys to multi-discrete arrays for movement.
    base_attack_names : dict
        Mapping of short string attack keys to multi-discrete arrays for attack.
    _base_actions : list of str
        Flattened list of "movement+attack" combos for indexing.
    action_idx_lookup : dict
        Maps a combo string (e.g. 'd+lp') to an integer index.
    input_lookup : dict
        Maps integer action indices back to multi-discrete arrays for the environment.
    move_pattern_names : dict
        Named movement patterns (e.g., quarter-circles, half-circles).
    agent_state : dict
        Stores per-agent info such as:
          - 'move_sequence': deque of action indices queued
          - 'character': str name of selected character
          - 'super_art': int representing chosen super art

    Methods
    -------
    reset(characters, super_arts)
        Clear any stored move sequences for each agent and store their chosen characters
        and super arts.
    in_sequence(player: str) -> bool
        Check if the specified player (agent) still has queued moves to execute.
    _combine_actions(move_string: str, attack_string: str) -> list of str
        Internal helper to produce a list of "move+attack" strings from standard patterns.
    _hold_direction(direction, min_frame, max_frame, release='') -> list of str
        Internal helper for charge-style moves by holding a direction for a random length
        of time, then optionally releasing an attack.
    _repeat_attack(attack_string, min_repeats, max_repeats, tap='') -> list of str
        Internal helper for repeated button presses (e.g., repeated punch).
    _raw_action_string(action_string: str) -> list of str
        Splits an action string by underscore, used for direct parsing.
    _decode_action_string(action_string: str) -> list of str
        Parses custom combo syntax into a final list of 'dir+attack' tokens.
    action_space_size() -> int
        Returns the total number of possible "movement+attack" combos in multi-discrete form.
    sample_character_special(player: str) -> list of str
        Generates a special or super-art combo for the given agent's character.
    sample_movement_action(prob_dash=0.15, repeat_min=12, repeat_max=64) -> list of str
        Samples movement patterns or repeated directional inputs (e.g. dashes).
    sample_jump_action(prob_super_jump=0.15) -> list of str
        Samples a jump sequence, optionally super jump if rand < prob_super_jump.
    string_to_idx(string_list: list) -> list of int
        Converts each 'dir+attack' token into an integer action index; handles KeyError fallback.
    sample(prob_jump=0.04, prob_basic=0.21, prob_combo=0.35, prob_cancel=0.20, prob_movement=0.30)
        Main entry point to produce the next action(s) for all agents. Each agent either continues
        a queued sequence or samples a new set of moves based on given probabilities. Returns a dict
        containing both discrete indices and multi-discrete arrays for each agent.
    """
    
    def __init__(self,
                 environment_name: str = 'sfiii',
                 mode: str = 'multi_discrete',
                 frame_skip: int = 4):
        """
        Initialize the ComboInjector.

        Parameters
        ----------
        environment_name : str, optional
            Name of the environment (default is 'sfiii').
        mode : str, optional
            Action mode (must be 'multi_discrete'); future expansion possible.
        frame_skip : int, optional
            Frame skip value that affects the duration of certain hold combos (default 4).
        """

        self.environment_name = environment_name
        self.mode = mode
        self.frame_skip = frame_skip

        # Only support multi_discrete for now
        if self.mode != 'multi_discrete':
            raise ValueError("Only 'multi_discrete' mode is currently supported.")
            
        # Characters and possible moves (as a nested dict)
        self.characters = {
            'sfiii': {
                "Alex" : {'power_bomb' : {'prob': 0.15, 'combo_str' : 'comb_hc_p'},
                          'spiral_ddt' : {'prob' : 0.15, 'combo_str' : 'comb_hc_k'},
                          'flash_chop' : {'prob' : 0.15, 'combo_str' : 'comb_qc_p'},
                          'air_knee_smash' : {'prob' : 0.15, 'combo_str' : 'comb_dp_k'},
                          'air_stampede' : {'prob' : 0.15, 'combo_str' : 'hold_d_16_64_k'},
                          'slash_elbow' : {'prob' : 0.15, 'combo_str' : 'hold_lr_16_64_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_fc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Twelve" : {'ndl' : {'prob': 0.3, 'combo_str' : 'comb_qc_p'},
                            'axe' : {'prob': 0.3, 'combo_str' : 'comb_qc_p/rep_p_3_12_t'},
                            'dra' : {'prob' : 0.3, 'combo_str' : 'comb_qc_k'},
                            'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Hugo" : {'shootdown_backbreaker' : {'prob': 0.15, 'combo_str' : 'comb_dp_k'},
                            'ultra_throw' : {'prob': 0.15, 'combo_str' : 'comb_hc_k'},
                            'moonsault_press' : {'prob' : 0.15, 'combo_str' : 'comb_fc_p'},
                            'meat_squasher' : {'prob' : 0.15, 'combo_str' : 'comb_fc_k'},
                            'giant_palm_bomber' : {'prob' : 0.15, 'combo_str' : 'comb_qc_p'},
                            'monster_lariat' : {'prob' : 0.15, 'combo_str' : 'comb_qc_k'},
                            'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2fc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp/rep_mp_0_8_',}
                          },
                "Sean" : {'zenten' : {'prob': 0.18, 'combo_str' : 'comb_qc_p'},
                          'sean_tackle' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p/rep_p_0_8_'},
                          'dragon_smash' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'tornado_ryuubi_kyaku' : {'prob' : 0.36, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp/rep_mp_0_12_t',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Makoto" : {'karakusa' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'hayate_oroshi' : {'prob' : 0.36, 'combo_str' : 'comb_qc_p/rep_p_0_8_'},
                          'fukiage' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'tsurugi' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Elena" : {'rhino_horn' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'mallet_smash' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p'},
                          'spin_scythe' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'scratch_wheel_lynx_tail' : {'prob' : 0.36, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mk',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Ibuki" : {'raida' : {'prob': 0.18, 'combo_str' : 'comb_hc_p'},
                          'kasumi_gake_tsumuji' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'tsuji_goe' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'kunai_kubi_ori' : {'prob' : 0.18, 'combo_str' : 'comb_qc_p'},
                          'kazekiri_hien' : {'prob' : 0.18, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/rep_mp_0_16_t',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Chun-Li" : {'kikoken' : {'prob': 0.18, 'combo_str' : 'comb_hc_p'},
                          'hazanshu' : {'prob' : 0.18, 'combo_str' : 'comb_hc_k'},
                          'spinning_bird_kick' : {'prob' : 0.18, 'combo_str' : 'hold_d_16_64_k'},
                          'hyakuretsu_kyaku' : {'prob' : 0.18, 'combo_str' : 'rep_k_3_16_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Dudley" : {'ducking_straight_short_swing_blow' : {'prob': 0.3, 'combo_str' : 'comb_hc_k/rep_p_0_4_t'},
                            'ducking_upper_short_swing_blow' : {'prob': 0.3, 'combo_str' : 'comb_hc_k/rep_k_0_4_t'},
                          'machine_gun_blow_cross_counter' : {'prob' : 0.3, 'combo_str' : 'comb_hc_p'},
                          'jet_upper' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp/rep_mp_3_12_t',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Necro" : {'snake_fang_rising_cobra' : {'prob': 0.18, 'combo_str' : 'comb_hc_k'},
                          'denji_blast' : {'prob' : 0.18, 'combo_str' : 'comb_dp_p/rep_p_0_12_t'},
                          'flying_viper' : {'prob' : 0.18, 'combo_str' : 'comb_qc_p'},
                          'rising_kobra' : {'prob' : 0.18, 'combo_str' : 'comb_qc_k'},
                          'tornado_hook' : {'prob' : 0.18, 'combo_str' : 'comb_hc_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/rep_mp_0_16_t',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Q" : {'capture_deadly_blow' : {'prob': 0.225, 'combo_str' : 'comb_hc_k'},
                          'dashing_head' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_p/rep_p_0_8_'},
                          'dashing_leg' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'high_speed_barage' : {'prob' : 0.225, 'combo_str' : 'comb_qc_p'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Oro" : {'niu_riki' : {'prob': 0.225, 'combo_str' : 'comb_hc_p'},
                          'nichirin_shou' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_p'},
                          'oni_yanma' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_p'},
                          'jinchuu_watari_hitobashira_nobori' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp/hold_lr_4_24_p',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Urien" : {'metallic_sphere' : {'prob': 0.225, 'combo_str' : 'comb_qc_p/rep_p_0_12_'},
                          'chariot_tackle' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'dangerous_headbutt' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_p'},
                          'violence_knee_drop' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_64_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Remy" : {'light_of_virtue' : {'prob': 0.225, 'combo_str' : 'hold_lr_16_64_p'},
                          'light_of_virtue_low' : {'prob' : 0.225, 'combo_str' : 'hold_lr_16_64_k'},
                          'rising_rage_flash' : {'prob' : 0.225, 'combo_str' : 'hold_d_16_128_k'},
                          'cold_blue_kick' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Ryu" : {'hadouken' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'shoryuken' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'joudan_sokutou_geri' : {'prob' : 0.225, 'combo_str' : 'comb_hc_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mp/rep_mp_0_16_',}
                          },
                "Gouki" : {'go_zankuu_hadouken' : {'prob': 0.15, 'combo_str' : 'comb_qc_p'},
                          'shakenutsu-hadouken' : {'prob' : 0.15, 'combo_str' : 'comb_hc_p'},
                          'go_shoryuken' : {'prob' : 0.15, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.15, 'combo_str' : 'comb_qc_k'},
                          'hyakkishu_go' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_p_0_2_t'},
                          'hyakkishu_sho' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_k_0_2_t'},
                          'hyakkishu_sho_sai' : {'prob' : 0.075, 'combo_str' : 'comb_dp_k/rep_mpk_0_2_t'},
                          'shun_goku_satsu' : {'prob' : 0.055, 'combo_str' : 'raw_+lp_+_+lp/comb_lr_/raw_+lk_+_+hp'},
                          'target_combo_1' : {'prob' : 0.02, 'combo_str' : 'rep_mp_5_5_t/raw_+/rep_hp_5_5_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Yun" : {'zenpou_tenshin' : {'prob': 0.225, 'combo_str' : 'comb_hc_k'},
                          'kobokushi_zesshou_hohou' : {'prob' : 0.225, 'combo_str' : 'comb_qc_p'},
                          'tetsuzanko' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'nishoukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mp',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
                "Yang" : {'tourou_zan_byakko_soushouda' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'senkyuutai' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'zenpou_tenshin' : {'prob' : 0.225, 'combo_str' : 'comb_hc_k'},
                          'kaihou' : {'prob' : 0.225, 'combo_str' : 'comb_dp_k'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk',
                                         'combo_str_3' : 'comb_2qc_mp',}
                          },
                "Ken" : {'hadouken' : {'prob': 0.225, 'combo_str' : 'comb_qc_p'},
                          'shoryuken' : {'prob' : 0.225, 'combo_str' : 'comb_dp_p'},
                          'tatsumaki_senpukyaku' : {'prob' : 0.225, 'combo_str' : 'comb_qc_k'},
                          'grab' : {'prob' : 0.225, 'combo_str' : 'rep_lpk_1_8_t'},
                          'super_art' : {'prob' : 0.1, 'combo_str_1' : 'comb_2qc_mp',
                                         'combo_str_2' : 'comb_2qc_mk/rep_mk_0_16_t',
                                         'combo_str_3' : 'comb_2qc_mk',}
                          },
            },
        }

        if self.mode != 'multi_discrete':
            raise NotImplementedError(
                f"Only 'multi_discrete' is supported, got mode={mode}"
            )

        # Base movement definitions (multi-discrete)
        self.base_movement_names = {
            '':   np.array([0, 0]),
            'l':  np.array([1, 0]),
            'ul': np.array([2, 0]),
            'u':  np.array([3, 0]),
            'ur': np.array([4, 0]),
            'r':  np.array([5, 0]),
            'dr': np.array([6, 0]),
            'd':  np.array([7, 0]),
            'dl': np.array([8, 0]),
        }

        # Base attack definitions (multi-discrete)
        self.base_attack_names = {
            '':    np.array([0, 0]),
            'lp':  np.array([0, 1]),
            'mp':  np.array([0, 2]),
            'hp':  np.array([0, 3]),
            'lk':  np.array([0, 4]),
            'mk':  np.array([0, 5]),
            'hk':  np.array([0, 6]),
            'lpk': np.array([0, 7]),
            'mpk': np.array([0, 8]),
            'hpk': np.array([0, 9]),
        }

        # Flatten the actions into a list of "movement+attack" combos for indexing
        self._base_actions = []
        for move_name, move_arr in self.base_movement_names.items():
            for attack_name, attack_arr in self.base_attack_names.items():
                combo_str = f'{move_name}+{attack_name}'
                self._base_actions.append(combo_str)
                
        # Map combo string -> index
        self.action_idx_lookup = {
            combo_str: i for i, combo_str in enumerate(self._base_actions)
        }

        # Reverse mapping: index -> multi-discrete array
        self.input_lookup = {}
        for combo_str, idx in self.action_idx_lookup.items():
            move_part, attack_part = combo_str.split('+')
            movement_arr = self.base_movement_names[move_part]
            attack_arr = self.base_attack_names[attack_part]
            self.input_lookup[idx] = (movement_arr + attack_arr).tolist()

        # Named movement patterns (quarter circles, half circles, etc.)
        self.move_pattern_names = {
            'rqc':  ['d', 'dr', 'r'],
            'lqc':  ['d', 'dl', 'l'],
            'rhc':  ['l', 'dl', 'd', 'dr', 'r'],
            'lhc':  ['r', 'dr', 'd', 'dl', 'l'],
            'rdp':  ['r', 'd', 'dr'],
            'ldp':  ['l', 'd', 'dl'],
            'rfc':  ['r', 'dr', 'd', 'dl', 'l', 'ul', 'u', 'ur'],
            'lfc':  ['l', 'dl', 'd', 'dr', 'r', 'ur', 'u', 'ul'],
            'sr':   ['r', '', 'r'],
            'sl':   ['l', '', 'l'],
            'sur':  ['d', 'ur'],
            'su':   ['d', 'u'],
            'sul':  ['d', 'ul'],
        }

        # Store per-agent state (character, super_art, move_sequence, etc.)
        self.agent_state = {}
        
    def reset(self, characters, super_arts):
        """
        Reset or initialize agent states.

        Parameters
        ----------
        characters : list of str
            A list of character names for each agent, e.g. ['Alex', 'Ken'].
        super_arts : list of int
            A list of super art indices for each agent, e.g. [1, 3].

        Raises
        ------
        NotImplementedError
            If the chosen character or super art is not supported.
        """
        self.agent_state = {}
        for i, (character, super_art) in enumerate(zip(characters, super_arts)):
            if character not in self.characters[self.environment_name]:
                raise NotImplementedError(f"Character '{character}' not supported yet.")
            if super_art not in [1, 2, 3]:
                raise NotImplementedError(f"Super art '{super_art}' not supported.")
            self.agent_state[f'agent_{i}'] = {'move_sequence' : deque(),
                                               'character' : character,
                                               'super_art' : super_art}
        
    def in_sequence(self, player: str) -> bool:
        """
        Check if the specified player's sequence is non-empty (still queued up moves).

        Parameters
        ----------
        player : str
            The agent key, e.g. 'agent_0' or 'agent_1'.

        Returns
        -------
        bool
            True if there are queued moves, False otherwise.
        """
        return len(self.agent_state[player]['move_sequence']) > 0
    
    def _combine_actions(self, move_string: str, attack_string: str) -> list:
        """
        Internal helper that stitches together a movement pattern with an attack.

        Parameters
        ----------
        move_string : str
            A short code indicating a special pattern (e.g. 'qc', '2qc', 'dp', 'lr') or
            literal direction.
        attack_string : str
            A code for the type of attack (e.g. 'p', 'k', 'lp').

        Returns
        -------
        list of str
            Each element is e.g. 'd+lp' or 'r+lk', forming a sequence.
        """
        
        # Parse the move_string to get a movement list
        if move_string == 'qc':
            m_seq = self.move_pattern_names[np.random.choice(['rqc', 'lqc'])]
        elif move_string == '2qc':
            m_seq = self.move_pattern_names[np.random.choice(['rqc', 'lqc'])] * 2
        elif move_string == 'hc':
            m_seq = self.move_pattern_names[np.random.choice(['rhc', 'lhc'])]
        elif move_string == 'fc':
            m_seq = self.move_pattern_names[np.random.choice(['rfc', 'lfc'])]
        elif move_string == '2fc':
            m_seq = self.move_pattern_names[np.random.choice(['rfc', 'lfc'])] * 2
        elif move_string == 'dp':
            m_seq = self.move_pattern_names[np.random.choice(['rdp', 'ldp'])]
        elif move_string == '2dp':
            m_seq = self.move_pattern_names[np.random.choice(['rdp', 'ldp'])] * 2
        elif move_string == 'lr':
            m_seq = [np.random.choice(['l', 'r'])]
        else:
            m_seq = [move_string]
        
        # Decide the final attack
        if attack_string == 'p':
            a = np.random.choice(['lp', 'mp', 'hp'])
        elif attack_string == 'k':
            a = np.random.choice(['lk', 'mk', 'hk'])
        else:
            a = attack_string

        # Align them so that the last move gets the attack
        a_seq = [''] * (len(m_seq) - 1) + [a]
        return [f"{m}+{atk}" for m, atk in zip(m_seq, a_seq)] 
    
    def _hold_direction(self, direction: str, min_frame: str, max_frame: str, release: str = '') -> list:
        """
        Internal helper for a "hold direction" move, e.g. hold down for N frames, then release.

        Parameters
        ----------
        direction : str
            e.g. 'd', 'lr' indicating the direction to hold.
        min_frame : str
            Minimum frames to hold (string from the combo_str).
        max_frame : str
            Maximum frames to hold (string from the combo_str).
        release : str, optional
            If 'p' or 'k', we pick a random punch/kick attack upon release. Otherwise literal.

        Returns
        -------
        list of str
            e.g. ['d+','d+','d+','u+lk'] if we hold down for 3 steps, then release with 'lk'.
        """
        
        min_f = int(min_frame)
        max_f = int(max_frame)
        
        # Determine the release attack if applicable
        if release == 'p':
            release = np.random.choice(['lp', 'mp', 'hp'])
        elif release == 'k':
            release = np.random.choice(['lk', 'mk', 'hk'])
            
        num_steps = np.random.randint(min_f // self.frame_skip, (max_f + 1) // self.frame_skip)
        
        # Build the direction sequence
        if direction == 'd':
            dir_held = np.random.choice(['d', 'dl', 'dr'])
            # If there's a release, append 'u' after
            # Example: hold [d, d, d], then 'u'
            m_seq = [dir_held] * num_steps
            if release:
                m_seq.append('u')
            a_seq = [''] * (len(m_seq) - bool(release)) + [release] * bool(release)
            
        elif direction == 'lr':
            dir_held = np.random.choice(['l', 'dl', 'dr', 'r'])
            m_seq = [dir_held] * num_steps  
            # If there's a release, we do the opposite direction or something else
            if release:
                m_seq.append('l' if dir_held.endswith('r') else 'r')
            a_seq = [''] * (len(m_seq) - bool(release)) + [release] * bool(release)
        
        else:
            # fallback
            m_seq = [direction]
            a_seq = ['']
            
        return [f"{m}+{atk}" for m, atk in zip(m_seq, a_seq)]
    
    def _repeat_attack(self, attack_string: str, min_repeats: str, max_repeats: str, tap: str = '') -> list:
        """
        Internal helper for repeated button presses (e.g., repeated punch or kick).

        Parameters
        ----------
        attack_string : str
            e.g. 'p', 'k', 'lp', 'mk'
        min_repeats : str
            Minimum number of repeats (string from combo_str).
        max_repeats : str
            Maximum number of repeats (string from combo_str).
        tap : str
            If not empty, means we insert a '+' after each attack to create a "tap" pattern.

        Returns
        -------
        list of str
            e.g. ['+lp', '+', '+lp', '+', '+lp'] for 3 repeated taps.
        """
        min_r = int(min_repeats)
        max_r = int(max_repeats)
        do_tap = bool(tap)
        
        # If the code is 'p' or 'k', pick a specific punch/kick
        if attack_string == 'p':
            attack_string = np.random.choice(['lp', 'mp', 'hp'])
        elif attack_string == 'k':
            attack_string = np.random.choice(['lk', 'mk', 'hk'])
            
        reps = np.random.randint(min_r, max_r + 1)
        seq = [f'+{attack_string}'] + ['+'] * do_tap
        return seq * reps
       
    def _decode_action_string(self, action_string: str) -> list:
        """
        Parse the custom combo syntax into final 'dir+attack' tokens.

        Each sub-string is separated by '/', and each chunk is something like:
            'comb_qc_p' or 'hold_d_16_64_k' or 'rep_p_0_8_t'.

        Parameters
        ----------
        action_string : str
            e.g. 'comb_qc_p/rep_p_0_8_t'

        Returns
        -------
        list of str
            Final list e.g. ['d+lp','dr+lp','r+lp'] after expansions.
        """
        action_sequence = []
        # A single action_string can have multiple segments delimited by '/'
        for sub_part  in action_string.split('/'):
            parts = sub_part.split('_')
            # e.g. parts might be ['comb', 'qc', 'p'] or ['hold','d','16','64','k']
            if parts[0] == 'comb':
                # comb_qc_p
                move_str, attack_str = parts[1], parts[2]
                action_sequence += self._combine_actions(move_str, attack_str)
            elif parts[0] == 'hold':
                # hold_direction
                direction, min_frame, max_frame, release = parts[1:]
                action_sequence += self._hold_direction(direction, min_frame, max_frame, release)
            elif parts[0] == 'rep':
                # repeated attack
                attack_str, min_r, max_r, tap_str = parts[1:]
                action_sequence += self._repeat_attack(attack_str, min_r, max_r, tap_str)
            elif parts[0] == 'raw':
                # raw_+lp => literal tokens
                action_sequence += parts[1:]
                
        return action_sequence
    
    def action_space_size(self) -> int:
        """
        Return the size of the known action space for multi-discrete combos.

        Returns
        -------
        int
            Number of possible "movement + attack" combos.
        """
        return len(self._base_actions)
        
    def sample_character_special(self, player: str) -> list:
        """
        Generates a special or super-art combo for the given agent's character.

        Parameters
        ----------
        player : str
            e.g. 'agent_0', referencing self.agent_state[player].

        Returns
        -------
        list of str
            A list of combos in the form ['d+lp', 'dr+lp', 'r+lp', ...] or fallback if not found.
        """
        
        character = self.agent_state[player]['character']
        super_art = self.agent_state[player]['super_art']
        
        if character not in self.characters[self.environment_name]:
            raise NotImplementedError(f"Character '{character}' not supported yet.")
            
        # Weighted random selection among that character's moves
        roll = np.random.rand()
        moves_dict = self.characters[self.environment_name][character]
        
        # We sum up probabilities to pick a move
        prob_acc = 0.0        
        for move_name, params in moves_dict.items():
            prob_acc += params['prob']
            if roll <= prob_acc:
                # If this is a super_art, we pick the appropriate suffix
                if move_name == 'super_art':
                    # e.g. 'combo_str_2'
                    suffix = f"_{super_art}"
                    combo_key = 'combo_str' + suffix
                    action_str = params[combo_key]
                    return self._decode_action_string(action_str)
                else:
                    # Normal move
                    action_str = params['combo_str']
                    return self._decode_action_string(action_str)
        # If none matched (sum of probabilities < 1?), fallback
        return [np.random.choice(self._base_actions)]
            
    def sample_movement_action(self,
                               prob_dash: float = 0.15,
                               repeat_min: int = 12,
                               repeat_max: int = 64) -> list:
        """
        Samples a movement pattern or repeated directional inputs (e.g. dashes).

        Parameters
        ----------
        prob_dash : float, optional
            Probability of a short dash pattern like ['r+','+','r+'] (default 0.15).
        repeat_min : int, optional
            Minimum frames for repeated direction (default 12).
        repeat_max : int, optional
            Maximum frames for repeated direction (default 64).

        Returns
        -------
        list of str
            e.g. ['r+', '+', 'r+'] or repeated directional combos.
        """
        roll = np.random.rand()
        if roll < prob_dash:
            move = np.random.choice(['r+', 'l+'])
            return [move, '+', move]
        
        # Otherwise repeated direction        
        repeated = np.random.choice(['l+', 'dl+', 'd+', 'dr+', 'r+'])
        length = np.random.randint(repeat_min//self.frame_skip, repeat_max//self.frame_skip)
        return [repeated] * length
    
    def sample_jump_action(self, prob_super_jump=0.15) -> list:
        roll = np.random.rand()
        # 15% chance to do some lateral movement, otherwise random repeated direction
        if roll < prob_super_jump:
            # super jump pattern
            moves = [
                ['d+', 'ul+'],
                ['d+', 'u+'],
                ['d+', 'ur+'],
            ]
            return moves[np.random.randint(len(moves))]
        else:
            # normal jump
            return [np.random.choice(['ul+', 'u+', 'ur+'])]

    def string_to_idx(self, string_list: list) -> list:
        """
        Convert each 'dir+attack' token into an integer action index.

        Parameters
        ----------
        string_list : list of str
            e.g. ['d+lp', 'dr+lp', 'r+lp'].

        Returns
        -------
        list of int
            The integer indices corresponding to each token.

        Notes
        -----
        If a token is not found in self.action_idx_lookup, a random fallback is used.
        """
        idx_list = []
        for s in string_list:
            try:
                idx_list.append(self.action_idx_lookup[s])
            except KeyError:
                # If something fails, we skip
                print(f"{s} not found in action lookup")
                idx_list.append(np.random.randint(0, len(self.action_idx_lookup)))
        return idx_list

    def sample(self,
               prob_jump: float = 0.04,
               prob_basic: float = 0.21,
               prob_combo: float = 0.35,
               prob_cancel: float = 0.2,
               prob_movement: float = 0.3):
        """
        Main entry point to produce the next action(s) for all agents.

        Each agent either continues any queued sequence or samples a new set of moves
        based on the given probabilities. The result is a dictionary containing both
        discrete indices and multi-discrete arrays for each agent.

        Probabilities:
          - prob_jump:    e.g. 0.04 chance for a jump action
          - prob_basic:   e.g. 0.21 chance for a single random basic action
          - prob_combo:   e.g. 0.35 chance to sample a special/super combo
          - prob_cancel:  e.g. 0.20 chance to truncate the combo early
          - prob_movement:e.g. 0.30 chance for a movement-based pattern

        Parameters
        ----------
        prob_jump : float, optional
            Probability to pick a jump action (default 0.04).
        prob_basic : float, optional
            Probability to pick a single random basic action (default 0.21).
        prob_combo : float, optional
            Probability to pick a special combo for the character (default 0.35).
        prob_cancel : float, optional
            Probability to truncate the combo (default 0.20).
        prob_movement : float, optional
            Probability to pick a movement-based action (default 0.30).

        Returns
        -------
        dict
            {
              'discrete': {'agent_0': int, 'agent_1': int, ...},
              'multi_discrete': {'agent_0': [x,y], 'agent_1': [x,y], ...}
            }

        Notes
        -----
        1. We sum (prob_jump + prob_basic + prob_combo + prob_movement) and normalize.
        2. For each agent, if no moves remain in their queue, we roll a random number
           and pick from the categories.
        3. Once a new sequence is sampled, we store it in the agent's 'move_sequence' deque.
        4. Finally, we pop one action from each agent's queue and return it.
        """
        actions = {'discrete' : {},
                   'multi_discrete' : {}}
        
        # Normalize probabilities
        raw_probs = np.array([prob_jump, prob_basic, prob_combo, prob_movement])
        raw_probs /= raw_probs.sum()  # ensure they sum to 1
        cdfs = np.cumsum(raw_probs)
        
        # For each agent, pick next action if needed
        for agent_id in self.agent_state:
            if not self.in_sequence(agent_id):
                roll = np.random.rand()
                # Decide which category
                if roll < cdfs[0]:
                    # Jump
                    seq_str = self.sample_jump_action()
                    seq_idx = self.string_to_idx(seq_str)
                    self.agent_state[agent_id]['move_sequence'] = deque(seq_idx)
                elif roll < cdfs[1]:
                    # Basic single action
                    single_idx = np.random.randint(0, len(self.action_idx_lookup))
                    self.agent_state[agent_id]['move_sequence'] = deque([single_idx])
                elif roll < cdfs[2]:
                    # Character-specific combos
                    seq_str = self.sample_character_special(agent_id)
                    seq_idx = self.string_to_idx(seq_str)
                    # Possibly cancel early
                    if np.random.rand() < prob_cancel:
                        cutoff = np.random.randint(1, len(seq_idx) + 1)
                        seq_idx = seq_idx[:cutoff]
                    self.agent_state[agent_id]['move_sequence'] = deque(seq_idx)
                else:
                    # Movement-based combos
                    move_seq_str = self.sample_movement_action()
                    seq_idx = self.string_to_idx(move_seq_str)
                    self.agent_state[agent_id]['move_sequence'] = deque(seq_idx)
                    
            # Pop the next action
            a_idx = self.agent_state[agent_id]['move_sequence'].popleft()
            a_multi = self.input_lookup[a_idx]
            
            actions['discrete'][agent_id] = a_idx
            actions['multi_discrete'][agent_id] = a_multi
        
        return actions
    