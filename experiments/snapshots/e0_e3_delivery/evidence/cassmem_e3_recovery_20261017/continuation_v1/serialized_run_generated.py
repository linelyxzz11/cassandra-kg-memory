def run(self):
    fixtures=json.loads((self.out/'fixtures.json').read_text(encoding='utf-8'));docs={};embeds={}
    for f in fixtures[:5]:
        r=base.GraphRecord(**{**f['versions'][0]['record'],'scope_id':self.scope})
        self.b.insert(self.cell,r);docs[r.memory_id]=r.rawerk;embeds[r.memory_id]=f['versions'][0]['vector']
    self.index.load_scope(self.scope,docs,embeds)
    self.service=SerializedUpdateService(self.b,self.cell,self.index)
    self.service.adopt(self.records[1],self.vec[1],known_relations={r for v in self.f['versions'] for r in v['required_relations']})
    other=next(f for f in fixtures[:5] if f['memory_id']!=self.mid)
    self.other_records={i+1:base.GraphRecord(**{**v['record'],'scope_id':self.scope}) for i,v in enumerate(other['versions'])}
    self.other_vectors={i+1:v['vector'] for i,v in enumerate(other['versions'])}
    self.service.adopt(self.other_records[1],self.other_vectors[1])
    self.v3_waiting=threading.Event();assert self.read('initial')
    def reader():
        while not self.stop.is_set():self.read('concurrent');self.stop.wait(.02)
    scenario=self.c['scenario']
    with ThreadPoolExecutor(4) as pool:
        reader_future=pool.submit(reader)
        try:
            if self.c['blocked_stage']:
                self.submit(2);f2=pool.submit(self.apply,2,pause=True)
                assert self.reached.wait(60);self.submit(3);f3=pool.submit(self.apply,3)
                assert self.v3_waiting.wait(60)
                # v3 has called into the service but cannot acquire this ID's lock.
                assert not f3.done();self.emit('timeline.jsonl','v3_wait_confirmed',version=3)
                if scenario=='different_id_progress':
                    assert pool.submit(self.apply,2,other=True).result(60)=='applied'
                    self.emit('timeline.jsonl','other_id_finished_while_target_blocked')
                self.read('while_v3_waiting');self.emit('timeline.jsonl','release_authorized',version=2);self.release.set()
                assert f2.result(60)=='applied';assert f3.result(60)=='applied'
            elif scenario=='retry_after_sparse':
                self.submit(3);assert self.apply(3,fail=True)=='InjectedFailure'
                self.read('after_injected_failure');self.submit(2);assert self.apply(2)=='stale'
                self.submit(3);assert self.apply(3)=='applied'
            elif scenario in ('reverse_arrival','duplicate','conflicting_duplicate'):
                self.submit(3);assert self.apply(3)=='applied'
                if scenario=='reverse_arrival':self.submit(2);assert self.apply(2)=='stale'
                elif scenario=='duplicate':self.submit(3);assert self.apply(3)=='duplicate'
                else:self.submit(3);assert self.apply(3,conflict=True)=='VersionConflict'
            else:
                self.submit(2);assert self.apply(2)=='applied';self.submit(3);assert self.apply(3)=='applied'
        finally:
            self.release.set();self.stop.set();reader_future.result(60)
    for i in range(3):self.read('final_'+str(i))
    self.emit('timeline.jsonl','case_complete')
