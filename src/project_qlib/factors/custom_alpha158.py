from __future__ import annotations

from qlib.contrib.data.handler import Alpha158


class Alpha158PlusOne(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return list(fields) + ["Ref($close, 1)/Ref($close, 5)-1"], list(names) + ["CSTM_MOM_5"]
